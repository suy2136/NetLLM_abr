"""Evaluate all public ABR inference modules with one official NetLLM LoRA."""

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ABR_ROOT.parent
RESULTS_ROOT = ABR_ROOT / 'artifacts' / 'results'
DEFAULT_BASE_MODEL = REPO_ROOT / 'downloaded_plms' / 'llama' / 'base'
DEFAULT_CHECKPOINT = ABR_ROOT / 'data' / 'ft_plms' / 'try_llama2_7b'
DEFAULT_EXP_POOL = ABR_ROOT / 'artifacts' / 'exp_pools' / 'exp_pool.pkl'
DEFAULT_OUTPUT = RESULTS_ROOT / 'official_lora_module_ablation.csv'


EXPERIMENTS = (
    {'name': 'netllm_original', 'temporal': 'none', 'token': 'none', 'spec': False},
    {'name': 'temporal_only', 'temporal': 'event-aware', 'token': 'none', 'spec': False},
    {'name': 'token_only', 'temporal': 'none', 'token': 'recent-timestep', 'spec': False},
    {'name': 'speculative_only', 'temporal': 'none', 'token': 'none', 'spec': True},
    {'name': 'temporal_token', 'temporal': 'event-aware', 'token': 'intra-timestep', 'spec': False},
    {'name': 'all_three', 'temporal': 'event-aware', 'token': 'intra-timestep', 'spec': True},
)


def validate_official_checkpoint(checkpoint_dir, rank):
    required = ('adapter_config.json', 'modules_except_plm.bin')
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if not any(
        (checkpoint_dir / name).is_file()
        for name in ('adapter_model.bin', 'adapter_model.safetensors')
    ):
        missing.append('adapter_model.bin or adapter_model.safetensors')
    if missing:
        raise FileNotFoundError(
            f'incomplete official NetLLM LoRA: {", ".join(missing)}'
        )
    config = json.loads(
        (checkpoint_dir / 'adapter_config.json').read_text(encoding='utf-8')
    )
    if str(config.get('peft_type', '')).upper() != 'LORA':
        raise ValueError('official checkpoint must use peft_type=LORA')
    configured_rank = config.get('r')
    if configured_rank is not None and int(configured_rank) != rank:
        raise ValueError(
            f'checkpoint rank does not match --rank: {configured_rank} != {rank}'
        )
    return config


def build_command(args, experiment):
    draft_steps = args.speculative_draft_steps if experiment['spec'] else 0
    command = [
        sys.executable, 'run_plm.py', '--test', '--fp16', '--seed', '1',
        '--plm-type', 'llama', '--plm-size', 'base', '--rank', str(args.rank),
        '--plm-dir', str(args.base_model_dir.resolve()),
        '--model-dir', str(args.checkpoint_dir.resolve()),
        '--exp-pool-path', str(args.exp_pool_path.resolve()),
        '--trace', args.trace, '--trace-num', str(args.trace_num),
        '--video', args.video, '--fixed-order',
        '--device', args.device, '--device-out', args.device,
        '--temporal-selector', experiment['temporal'],
        '--token-selector', experiment['token'],
        '--selector-history-steps', str(args.selector_history_steps),
        '--event-max-events', str(args.event_max_events),
        '--event-min-spacing', str(args.event_min_spacing),
        '--event-throughput-threshold', str(args.throughput_threshold),
        '--event-buffer-threshold', str(args.buffer_threshold),
        '--event-bitrate-jump-threshold', str(args.bitrate_jump_threshold),
        '--speculative-draft-steps', str(draft_steps),
        '--speculative-verification-mode', args.verification_mode,
        '--speculative-buffer-tolerance', str(args.buffer_tolerance),
        '--speculative-state-tolerance', str(args.state_tolerance),
        '--speculative-return-tolerance', str(args.return_tolerance),
    ]
    return command


def metrics_match(metrics, experiment, args):
    expected_steps = args.speculative_draft_steps if experiment['spec'] else 0
    return all((
        metrics.get('temporal_selector', 'none') == experiment['temporal'],
        metrics.get('selector', 'none') == experiment['token'],
        int(metrics.get('speculative_draft_steps', 0)) == expected_steps,
        metrics.get('inference_calls', 0) > 0,
    ))


def newest_matching_metrics(started_at, experiment, args):
    candidates = [
        path for path in RESULTS_ROOT.rglob('selector_metrics.json')
        if path.stat().st_mtime >= started_at
    ]
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime,
                       reverse=True):
        try:
            metrics = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if metrics_match(metrics, experiment, args):
            return path, metrics
    raise RuntimeError(
        f'{experiment["name"]} produced no matching selector_metrics.json'
    )


def scalar_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def add_baseline_comparisons(rows):
    baseline = next(
        (row for row in rows if row['experiment'] == 'netllm_original'), None
    )
    if baseline is None:
        return rows
    base_reward = baseline.get('mean_reward')
    base_latency = baseline.get('inference_latency_mean_ms')
    base_calls = baseline.get('target_plm_calls')
    for row in rows:
        reward = row.get('mean_reward')
        latency = row.get('inference_latency_mean_ms')
        calls = row.get('target_plm_calls')
        if all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (reward, base_reward)):
            row['mean_reward_delta_vs_netllm'] = reward - base_reward
            row['mean_reward_change_ratio_vs_netllm'] = (
                0.0 if base_reward == 0 else reward / base_reward - 1.0
            )
        if all(isinstance(value, (int, float)) and value > 0
               for value in (latency, base_latency)):
            row['inference_speedup_vs_netllm'] = base_latency / latency
            row['latency_reduction_vs_netllm'] = 1.0 - latency / base_latency
        if all(isinstance(value, (int, float)) and value > 0
               for value in (calls, base_calls)):
            row['target_plm_call_reduction_vs_netllm'] = 1.0 - calls / base_calls
    return rows


def run_signature(args):
    return {
        'checkpoint_dir': str(args.checkpoint_dir.resolve()),
        'base_model_dir': str(args.base_model_dir.resolve()),
        'exp_pool_path': str(args.exp_pool_path.resolve()),
        'rank': args.rank, 'seed': 1, 'trace': args.trace,
        'trace_num': args.trace_num, 'video': args.video,
        'selector_history_steps': args.selector_history_steps,
        'event_max_events': args.event_max_events,
        'event_min_spacing': args.event_min_spacing,
        'throughput_threshold': args.throughput_threshold,
        'buffer_threshold': args.buffer_threshold,
        'bitrate_jump_threshold': args.bitrate_jump_threshold,
        'speculative_draft_steps': args.speculative_draft_steps,
        'verification_mode': args.verification_mode,
        'buffer_tolerance': args.buffer_tolerance,
        'state_tolerance': args.state_tolerance,
        'return_tolerance': args.return_tolerance,
        'experiments': [item['name'] for item in EXPERIMENTS],
    }


def load_resume_rows(output, signature):
    rows_path = output.with_suffix('.json')
    manifest_path = output.with_suffix('.manifest.json')
    if not rows_path.is_file() and not manifest_path.is_file():
        return []
    if not rows_path.is_file() or not manifest_path.is_file():
        raise RuntimeError('resume requires both result JSON and manifest JSON')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('signature') != signature:
        raise ValueError('existing result manifest does not match this run')
    return json.loads(rows_path.read_text(encoding='utf-8'))


def write_results(rows, output, signature):
    rows = add_baseline_comparisons(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        'experiment', 'seed', 'mean_reward',
        'mean_reward_delta_vs_netllm',
        'mean_reward_change_ratio_vs_netllm', 'qoe_raw_mean',
        'mean_bitrate_mbps', 'total_rebuffer_s',
        'mean_rebuffer_s_per_chunk', 'mean_smoothness_mbps',
        'inference_latency_mean_ms', 'inference_latency_p50_ms',
        'inference_latency_p95_ms', 'inference_speedup_vs_netllm',
        'latency_reduction_vs_netllm', 'original_tokens_mean',
        'selected_tokens_mean', 'token_reduction_ratio',
        'temporal_history_reduction_ratio', 'intra_token_reduction_ratio',
        'acceptance_rate', 'target_plm_calls', 'llm_call_reduction_ratio',
        'target_plm_call_reduction_vs_netllm', 'fallback_calls', 'time',
        'metrics_path',
    ]
    fields_present = {key for row in rows for key in row}
    fields = [key for key in preferred if key in fields_present]
    fields.extend(sorted(fields_present.difference(fields)))
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix('.json').write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding='utf-8'
    )
    output.with_suffix('.manifest.json').write_text(
        json.dumps({
            'signature': signature,
            'completed_experiments': [row['experiment'] for row in rows],
            'requested_experiments': [item['name'] for item in EXPERIMENTS],
        }, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--exp-pool-path', type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--selector-history-steps', type=int, default=5)
    parser.add_argument('--event-max-events', type=int, default=3)
    parser.add_argument('--event-min-spacing', type=int, default=2)
    parser.add_argument('--throughput-threshold', type=float, default=0.60)
    parser.add_argument('--buffer-threshold', type=float, default=6.0)
    parser.add_argument('--bitrate-jump-threshold', type=int, default=1)
    parser.add_argument('--speculative-draft-steps', type=int, default=3)
    parser.add_argument('--verification-mode', choices=('greedy', 'sample'),
                        default='sample')
    parser.add_argument('--buffer-tolerance', type=float, default=1.0)
    parser.add_argument('--state-tolerance', type=float, default=0.25)
    parser.add_argument('--return-tolerance', type=float, default=0.01)
    parser.add_argument('--trace', default='fcc-test')
    parser.add_argument('--trace-num', type=int, default=100)
    parser.add_argument('--video', default='video1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--only', nargs='*',
                        choices=[item['name'] for item in EXPERIMENTS])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    positive = (
        args.rank, args.trace_num, args.selector_history_steps,
        args.event_min_spacing,
        args.throughput_threshold, args.buffer_threshold,
        args.bitrate_jump_threshold, args.speculative_draft_steps,
    )
    if any(value <= 0 for value in positive):
        parser.error(
            'rank, trace count, selector, event, and draft parameters '
            'must be positive'
        )
    if not 1 <= args.speculative_draft_steps <= 5:
        parser.error('--speculative-draft-steps must be between 1 and 5')
    if args.event_max_events < 0:
        parser.error('--event-max-events must be non-negative')
    if any(value < 0 for value in (
        args.buffer_tolerance, args.state_tolerance, args.return_tolerance,
    )):
        parser.error('speculative tolerances must be non-negative')
    return args


def main(argv=None):
    args = parse_args(argv)
    selected = [
        item for item in EXPERIMENTS
        if not args.only or item['name'] in args.only
    ]
    if args.dry_run:
        for experiment in selected:
            print(f"[{experiment['name']}] {shlex.join(build_command(args, experiment))}")
        return 0

    validate_official_checkpoint(args.checkpoint_dir, args.rank)
    if not (args.base_model_dir / 'config.json').is_file():
        raise FileNotFoundError(f'base model not found: {args.base_model_dir}')
    if not args.exp_pool_path.is_file():
        raise FileNotFoundError(f'experience pool not found: {args.exp_pool_path}')

    signature = run_signature(args)
    rows = load_resume_rows(args.output, signature) if args.resume else []
    completed = {row['experiment'] for row in rows}
    for experiment in selected:
        if experiment['name'] in completed:
            print(f"[{experiment['name']}] already complete; skipping", flush=True)
            continue
        command = build_command(args, experiment)
        print(f"[{experiment['name']}] {shlex.join(command)}", flush=True)
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path, metrics = newest_matching_metrics(
            started_at, experiment, args
        )
        rows.append({
            'experiment': experiment['name'], 'seed': 1,
            'checkpoint_dir': str(args.checkpoint_dir.resolve()),
            'metrics_path': str(metrics_path.resolve()),
            **scalar_metrics(metrics),
        })
        write_results(rows, args.output, signature)
    print(f'Results saved at: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
