"""Smoke-test the ABR token selector and MPC speculative decoding.

The official NetLLM ABR checkpoint is a LoRA adapter, not a standalone model.
Real mode therefore needs both its extracted checkpoint directory and the
separately downloaded Llama2-7B base model.

Examples:
  python analysis/smoke_test_inference_features.py --mode auto
  python analysis/smoke_test_inference_features.py --mode real \
      --checkpoint-dir data/ft_plms/try_llama2_7b \
      --device cuda:0
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / 'artifacts' / 'results'
DEFAULT_CHECKPOINT = ABR_ROOT / 'data' / 'ft_plms' / 'try_llama2_7b'
DEFAULT_BASE_MODEL = ABR_ROOT.parent / 'downloaded_plms' / 'llama' / 'base'
VP_CHECKPOINT_CANDIDATES = (DEFAULT_CHECKPOINT,)


def _valid_base_model(path):
    return path.is_dir() and (path / 'config.json').is_file()


def base_model_candidates(vp_checkpoint_candidates=VP_CHECKPOINT_CANDIDATES):
    """Return likely locations, including the base recorded by the VP adapter."""
    candidates = []
    for checkpoint_dir in vp_checkpoint_candidates:
        adapter_config = checkpoint_dir / 'adapter_config.json'
        if not adapter_config.is_file():
            continue
        try:
            with adapter_config.open(encoding='utf-8') as stream:
                recorded_path = json.load(stream).get('base_model_name_or_path')
        except (OSError, ValueError):
            continue
        if recorded_path:
            recorded_path = Path(recorded_path).expanduser()
            if recorded_path.is_absolute():
                candidates.append(recorded_path)

    workspace_root = ABR_ROOT.parents[1]
    project_root = ABR_ROOT.parent
    for root in (project_root, workspace_root):
        candidates.extend((
            root / 'downloaded_plms' / 'llama' / 'base',
            root / 'downloaded_plms' / 'llama2' / 'base',
        ))

    hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
    candidates.extend(sorted(
        hf_cache.glob('models--meta-llama--Llama-2-7b-hf/snapshots/*')
    ))

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def discover_base_model(explicit_path=None, vp_checkpoint_candidates=VP_CHECKPOINT_CANDIDATES):
    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    candidates = base_model_candidates(vp_checkpoint_candidates)
    return next((path for path in candidates if _valid_base_model(path)), DEFAULT_BASE_MODEL)


def checkpoint_problems(checkpoint_dir, base_model_dir, nbs_v19=False):
    checkpoint_dir = Path(checkpoint_dir)
    base_model_dir = Path(base_model_dir)
    problems = []
    if not base_model_dir.is_dir():
        problems.append(f'base model directory not found: {base_model_dir}')
    elif not (base_model_dir / 'config.json').is_file():
        problems.append(f'base model config.json not found: {base_model_dir}')

    if not checkpoint_dir.is_dir():
        problems.append(f'LoRA checkpoint directory not found: {checkpoint_dir}')
        return problems
    for filename in ('adapter_config.json', 'modules_except_plm.bin'):
        if not (checkpoint_dir / filename).is_file():
            problems.append(f'checkpoint file not found: {checkpoint_dir / filename}')
    adapter_weights = (
        checkpoint_dir / 'adapter_model.safetensors',
        checkpoint_dir / 'adapter_model.bin',
    )
    if not any(path.is_file() for path in adapter_weights):
        problems.append(
            'LoRA adapter weights not found (expected adapter_model.safetensors '
            f'or adapter_model.bin in {checkpoint_dir})'
        )
    if nbs_v19:
        for filename in ('nash_rank_allocator.pt', 'checkpoint_metadata.json'):
            if not (checkpoint_dir / filename).is_file():
                problems.append(
                    f'NBS checkpoint file not found: {checkpoint_dir / filename}'
                )
    return problems


def common_command(args):
    command = [
        sys.executable, 'run_plm.py', '--test',
        '--plm-type', 'llama', '--plm-size', 'base',
        '--rank', str(args.rank),
        '--fp16',
        '--plm-dir', str(Path(args.base_model_dir).resolve()),
        '--model-dir', str(Path(args.checkpoint_dir).resolve()),
        '--device', args.device, '--device-out', args.device,
        '--trace-num', '1', '--fixed-order',
    ]
    if args.nbs_v19:
        command.extend([
            '--nbs-v19', '--nbs-rank-budget', str(args.nbs_rank_budget),
            '--nbs-rank-config', str(args.nbs_rank_config),
        ])
    return command


def feature_command(feature, args):
    command = common_command(args)
    if feature == 'selector':
        command.extend([
            '--token-selector', 'recent-timestep',
            '--selector-history-steps', str(args.selector_history_steps),
            '--speculative-draft-steps', '0',
        ])
    elif feature == 'speculative':
        command.extend([
            '--token-selector', 'none',
            '--speculative-draft-steps', str(args.speculative_draft_steps),
            '--speculative-verification-mode', 'greedy',
        ])
    elif feature in ('temporal', 'intra', 'hierarchical',
                     'hierarchical-speculative'):
        temporal_enabled = feature in ('temporal', 'hierarchical',
                                       'hierarchical-speculative')
        intra_enabled = feature in ('intra', 'hierarchical',
                                    'hierarchical-speculative')
        command.extend([
            '--temporal-selector', (
                'event-aware' if temporal_enabled else 'none'
            ),
            '--token-selector', (
                'intra-timestep' if intra_enabled else 'none'
            ),
            '--event-max-events', str(args.event_max_events),
            '--speculative-draft-steps', (
                str(args.speculative_draft_steps)
                if feature == 'hierarchical-speculative' else '0'
            ),
        ])
        if feature == 'hierarchical-speculative':
            command.extend(['--speculative-verification-mode', 'greedy'])
    else:
        raise ValueError(f'unknown feature: {feature}')
    return command


def newest_metrics(since):
    candidates = [
        path for path in RESULTS_ROOT.rglob('selector_metrics.json')
        if path.stat().st_mtime >= since
    ]
    if not candidates:
        raise RuntimeError('run completed but selector_metrics.json was not produced')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_metrics(feature, metrics):
    if metrics.get('inference_calls', 0) <= 0:
        raise RuntimeError(f'{feature}: no inference call was recorded')
    if feature == 'selector':
        if metrics.get('selector') != 'recent-timestep':
            raise RuntimeError('selector: recent-timestep was not enabled')
        if metrics.get('selected_tokens_mean', float('inf')) > metrics.get(
            'original_tokens_mean', 0
        ):
            raise RuntimeError('selector: selected token count increased')
    elif feature == 'speculative':
        if metrics.get('speculative_draft_steps', 0) <= 0:
            raise RuntimeError('speculative: draft generation was not enabled')
        if metrics.get('draft_attempts', 0) <= 0 or metrics.get('drafted_actions', 0) <= 0:
            raise RuntimeError('speculative: MPC draft path was not exercised')
    else:
        temporal_expected = feature in (
            'temporal', 'hierarchical', 'hierarchical-speculative'
        )
        intra_expected = feature in (
            'intra', 'hierarchical', 'hierarchical-speculative'
        )
        if temporal_expected and (
            metrics.get('temporal_selector') != 'event-aware'
            or metrics.get('temporal_selector_calls', 0) <= 0
        ):
            raise RuntimeError(f'{feature}: temporal selector was not exercised')
        if intra_expected and (
            metrics.get('selector') != 'intra-timestep'
            or metrics.get('token_selector_calls', 0) <= 0
        ):
            raise RuntimeError(f'{feature}: intra-token selector was not exercised')
        if feature == 'hierarchical-speculative' and (
            metrics.get('draft_attempts', 0) <= 0
            or metrics.get('drafted_actions', 0) <= 0
        ):
            raise RuntimeError(
                'hierarchical-speculative: MPC draft path was not exercised'
            )


def run_real(args):
    summaries = {}
    for feature in args.features:
        command = feature_command(feature, args)
        print(f'[{feature}]', subprocess.list2cmdline(command), flush=True)
        started = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(started)
        with metrics_path.open(encoding='utf-8') as stream:
            metrics = json.load(stream)
        validate_metrics(feature, metrics)
        summaries[feature] = {
            'metrics_path': str(metrics_path),
            'mean_reward': metrics.get('mean_reward'),
            'token_reduction_ratio': metrics.get('token_reduction_ratio'),
            'temporal_history_reduction_ratio': metrics.get(
                'temporal_history_reduction_ratio'
            ),
            'intra_token_reduction_ratio': metrics.get(
                'intra_token_reduction_ratio'
            ),
            'draft_attempts': metrics.get('draft_attempts'),
            'acceptance_rate': metrics.get('acceptance_rate'),
            'llm_call_reduction_ratio': metrics.get('llm_call_reduction_ratio'),
        }
        print(f'[{feature}] PASS: {json.dumps(summaries[feature], indent=2)}')
    return summaries


def run_mock():
    modules = (
        'tests.test_selection_layout',
        'tests.test_selectors_torch',
        'tests.test_mpc_draft',
        'tests.test_speculative_acceptance',
    )
    command = [sys.executable, '-m', 'unittest', '-v', *modules]
    print('[mock]', subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ABR_ROOT, check=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('auto', 'check', 'mock', 'real'), default='auto')
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--base-model-dir', type=Path,
        help='base Llama directory; omitted means auto-detect the model used by VP',
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--nbs-v19', action='store_true')
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--nbs-rank-budget', type=int, default=512)
    parser.add_argument(
        '--nbs-rank-config', default='configs/nbs_v19_rank_config.json'
    )
    parser.add_argument('--selector-history-steps', type=int, default=5)
    parser.add_argument('--speculative-draft-steps', type=int, default=2)
    parser.add_argument('--event-max-events', type=int, default=3)
    parser.add_argument(
        '--features', nargs='+',
        choices=(
            'selector', 'speculative', 'temporal', 'intra', 'hierarchical',
            'hierarchical-speculative',
        ),
        default=['selector', 'speculative'],
    )
    args = parser.parse_args(argv)
    if args.selector_history_steps <= 0:
        parser.error('--selector-history-steps must be positive')
    if not 1 <= args.speculative_draft_steps <= 5:
        parser.error('--speculative-draft-steps must be between 1 and 5')
    if args.event_max_events < 0:
        parser.error('--event-max-events must be non-negative')
    if args.rank <= 0 or args.nbs_rank_budget <= 0:
        parser.error('rank and NBS rank budget must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.base_model_dir = discover_base_model(args.base_model_dir)
    if _valid_base_model(args.base_model_dir):
        print(f'Base model selected: {args.base_model_dir}')
    problems = checkpoint_problems(
        args.checkpoint_dir, args.base_model_dir, nbs_v19=args.nbs_v19
    )
    if problems:
        print('Real-model prerequisites: NOT READY')
        for problem in problems:
            print(f'  - {problem}')
    else:
        print('Real-model prerequisites: READY')

    if args.mode == 'check':
        return 1 if problems else 0
    if args.mode == 'real':
        if problems:
            print('Cannot run --mode real until the prerequisites above are installed.')
            return 2
        run_real(args)
    elif args.mode == 'mock' or problems:
        if args.mode == 'auto' and problems:
            print('Falling back to dependency-light mock smoke tests.')
        run_mock()
    else:
        run_real(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
