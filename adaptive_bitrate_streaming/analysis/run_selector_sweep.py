"""Run and summarize the ABR selector history-length sweep.

All arguments after ``--`` are forwarded to ``run_plm.py``.  For example:

    python analysis/run_selector_sweep.py -- --model-dir PATH --device cuda:0
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / 'artifacts' / 'results'


def configurations(history_steps):
    yield 'none', None
    for steps in history_steps:
        yield 'recent-timestep', steps


def command_for(selector, steps, forwarded):
    command = [sys.executable, 'run_plm.py', '--test', *forwarded]
    command.extend(['--token-selector', selector])
    if steps is not None:
        command.extend(['--selector-history-steps', str(steps)])
    return command


def newest_metrics(selector, steps):
    tag = 'selector_none' if selector == 'none' else f'selector_recent_timestep_h{steps}'
    candidates = [
        path for path in RESULTS_ROOT.rglob('selector_metrics.json')
        if tag in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(f'no metrics produced for {tag}')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-steps', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--output-csv', default='artifacts/results/selector_sweep.csv')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('forwarded', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.forwarded[1:] if args.forwarded[:1] == ['--'] else args.forwarded
    forbidden = {'--token-selector', '--selector-history-steps'}
    if forbidden.intersection(forwarded):
        parser.error('selector options are controlled by the sweep script')
    if any(steps <= 0 for steps in args.history_steps):
        parser.error('--history-steps values must be positive')

    rows = []
    for selector, steps in configurations(args.history_steps):
        command = command_for(selector, steps, forwarded)
        print(' '.join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(selector, steps)
        with metrics_path.open() as f:
            row = json.load(f)
        row['metrics_path'] = str(metrics_path)
        rows.append(row)

    if args.dry_run:
        return
    output_path = ABR_ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'selector', 'selector_history_steps', 'mean_reward',
        'inference_latency_mean_ms', 'inference_latency_p95_ms',
        'original_tokens_mean', 'selected_tokens_mean',
        'token_reduction_ratio', 'inference_calls', 'time', 'metrics_path',
    ]
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    main()
