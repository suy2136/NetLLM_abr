"""Compare ordinary ABR inference with robust-MPC draft horizons K=1..4."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / 'artifacts' / 'results'


def configurations(draft_steps):
    yield 0
    yield from draft_steps


def command_for(steps, mode, buffer_tolerance, state_tolerance, return_tolerance, forwarded):
    command = [sys.executable, 'run_plm.py', '--test', *forwarded]
    command.extend(['--speculative-draft-steps', str(steps)])
    if steps > 0:
        command.extend(['--speculative-verification-mode', mode])
        command.extend(['--speculative-buffer-tolerance', str(buffer_tolerance)])
        command.extend(['--speculative-state-tolerance', str(state_tolerance)])
        command.extend(['--speculative-return-tolerance', str(return_tolerance)])
    return command


def newest_metrics(steps, mode, buffer_tolerance, state_tolerance, return_tolerance):
    tag = (
        'speculative_none' if steps == 0
        else f'speculative_mpc_k{steps}_{mode}_btol{buffer_tolerance}_stol{state_tolerance}_rtol{return_tolerance}'
    )
    candidates = [
        path for path in RESULTS_ROOT.rglob('selector_metrics.json')
        if tag in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(f'no metrics produced for {tag}')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft-steps', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--verification-mode', choices=('greedy', 'sample'), default='sample')
    parser.add_argument('--buffer-tolerance', type=float, default=1.0)
    parser.add_argument('--state-tolerance', type=float, default=0.25)
    parser.add_argument('--return-tolerance', type=float, default=0.01)
    parser.add_argument('--output-csv', default='artifacts/results/speculative_sweep.csv')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('forwarded', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.forwarded[1:] if args.forwarded[:1] == ['--'] else args.forwarded
    forbidden = {
        '--speculative-draft-steps', '--speculative-verification-mode',
        '--speculative-buffer-tolerance',
        '--speculative-state-tolerance', '--speculative-return-tolerance',
    }
    if forbidden.intersection(forwarded):
        parser.error('speculative options are controlled by the sweep script')
    if any(steps <= 0 for steps in args.draft_steps):
        parser.error('--draft-steps values must be positive')
    if any(value < 0 for value in (
        args.buffer_tolerance, args.state_tolerance, args.return_tolerance
    )):
        parser.error('speculative tolerances must be non-negative')

    rows = []
    for steps in configurations(args.draft_steps):
        command = command_for(
            steps, args.verification_mode, args.buffer_tolerance,
            args.state_tolerance, args.return_tolerance, forwarded
        )
        print(' '.join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(
            steps, args.verification_mode, args.buffer_tolerance,
            args.state_tolerance, args.return_tolerance
        )
        with metrics_path.open() as f:
            row = json.load(f)
        row['metrics_path'] = str(metrics_path)
        rows.append(row)

    if args.dry_run:
        return
    output_path = ABR_ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'speculative_draft_steps', 'mean_reward',
        'inference_latency_mean_ms', 'inference_latency_p95_ms',
        'target_plm_calls', 'llm_call_reduction_ratio', 'acceptance_rate',
        'draft_attempts', 'drafted_actions', 'accepted_actions',
        'corrected_actions', 'queued_actions_served',
        'state_mismatch_fallbacks', 'buffer_mismatch_fallbacks',
        'feature_mismatch_fallbacks', 'return_mismatch_fallbacks',
        'throughput_predictor_updates', 'draft_generation_failures',
        'token_reduction_ratio', 'time', 'metrics_path',
    ]
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    main()
