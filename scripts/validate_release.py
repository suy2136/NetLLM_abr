"""Validate the pruned public repository and optionally run a GPU smoke test."""

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ABR_ROOT = REPO_ROOT / 'adaptive_bitrate_streaming'
DEFAULT_BASE_MODEL = REPO_ROOT / 'downloaded_plms' / 'llama' / 'base'
DEFAULT_CHECKPOINT = ABR_ROOT / 'data' / 'ft_plms' / 'try_llama2_7b'


def validate_scope(allow_upstream_data=False):
    problems = []
    excluded = [
        REPO_ROOT / 'viewport_prediction',
        REPO_ROOT / 'cluster_job_scheduling',
    ]
    if not allow_upstream_data:
        excluded.extend((
            ABR_ROOT / 'data' / 'all_models',
            ABR_ROOT / 'data' / 'traces' / 'train',
            ABR_ROOT / 'data' / 'traces' / 'valid',
            ABR_ROOT / 'data' / 'videos' / 'video2_sizes',
        ))
    for path in excluded:
        if path.exists():
            problems.append(f'excluded release path still exists: {path}')
    required = (
        REPO_ROOT / 'LICENSE', REPO_ROOT / 'NOTICE',
        REPO_ROOT / 'requirements-inference.txt',
        ABR_ROOT / 'artifacts' / 'exp_pools' / 'exp_pool.pkl',
        ABR_ROOT / 'plm_special' / 'models' / 'event_selection.py',
        ABR_ROOT / 'plm_special' / 'models' / 'selectors.py',
        ABR_ROOT / 'plm_special' / 'speculative' / 'mpc_draft.py',
    )
    for path in required:
        if not path.is_file():
            problems.append(f'required release file is missing: {path}')
    trace_dir = ABR_ROOT / 'data' / 'traces' / 'test' / 'fcc-test'
    trace_count = len(list(trace_dir.glob('*'))) if trace_dir.is_dir() else 0
    if trace_count < 100:
        problems.append(f'expected at least 100 FCC test files, found {trace_count}')
    video_dir = ABR_ROOT / 'data' / 'videos' / 'video1_sizes'
    if any(not (video_dir / f'video_size_{index}').is_file()
           for index in range(6)):
        problems.append('the six video1 size files are incomplete')
    return problems


def run(command, cwd=REPO_ROOT):
    print('+', subprocess.list2cmdline([str(item) for item in command]), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-model', action='store_true')
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--allow-upstream-data', action='store_true',
        help='allow locally restored upstream train/valid/video2/baseline data',
    )
    args = parser.parse_args()

    problems = validate_scope(args.allow_upstream_data)
    if problems:
        for problem in problems:
            print(f'- {problem}')
        return 1

    run([
        sys.executable, '-m', 'compileall', '-q',
        str(ABR_ROOT / 'analysis'), str(ABR_ROOT / 'plm_special'),
        str(REPO_ROOT / 'scripts'),
    ])
    run([
        sys.executable, '-m', 'unittest', 'discover',
        '-s', str(ABR_ROOT / 'tests'), '-p', 'test_*.py',
    ])
    if args.with_model:
        common = [
            '--base-model-dir', str(args.base_model_dir.resolve()),
            '--checkpoint-dir', str(args.checkpoint_dir.resolve()),
        ]
        run([
            sys.executable, str(REPO_ROOT / 'scripts' / 'check_installation.py'),
            *common, '--require-cuda',
        ])
        run([
            sys.executable,
            str(ABR_ROOT / 'analysis' / 'smoke_test_inference_features.py'),
            '--mode', 'real', *common, '--device', args.device,
            '--features', 'selector', 'speculative', 'hierarchical',
            'hierarchical-speculative',
        ], cwd=ABR_ROOT)
    print('Public ABR release validation: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
