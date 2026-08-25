"""Run an event-aware temporal-selector parameter sweep.

All arguments after ``--`` are forwarded to ``run_plm.py``.  The sweep keeps
the token selector and speculative inference disabled so that it measures only
temporal history selection.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / "artifacts" / "results"


def configurations(max_events):
    yield "none", 0
    for count in max_events:
        yield "event-aware", count


def command_for(selector, count, spacing, throughput, buffer, bitrate, forwarded):
    command = [sys.executable, "run_plm.py", "--test", *forwarded]
    command.extend([
        "--temporal-selector", selector,
        "--token-selector", "none",
        "--speculative-draft-steps", "0",
    ])
    if selector == "event-aware":
        command.extend([
            "--event-max-events", str(count),
            "--event-min-spacing", str(spacing),
            "--event-throughput-threshold", str(throughput),
            "--event-buffer-threshold", str(buffer),
            "--event-bitrate-jump-threshold", str(bitrate),
        ])
    return command


def selector_tag(selector, count, spacing, throughput, buffer, bitrate):
    if selector == "none":
        return "selector_none"
    return (
        f"temporal_event_aware_k{count}_spacing{spacing}"
        f"_tp{throughput:g}_buf{buffer:g}_br{bitrate}_token_none"
    )


def newest_metrics(selector, count, spacing, throughput, buffer, bitrate):
    tag = selector_tag(
        selector, count, spacing, throughput, buffer, bitrate
    )
    candidates = [
        path for path in RESULTS_ROOT.rglob("selector_metrics.json")
        if tag in path.parts and "speculative_none" in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(f"no metrics produced for {tag}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-events", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    parser.add_argument("--min-spacing", type=int, default=2)
    parser.add_argument("--throughput-threshold", type=float, default=0.60)
    parser.add_argument("--buffer-threshold", type=float, default=6.0)
    parser.add_argument("--bitrate-jump-threshold", type=int, default=1)
    parser.add_argument(
        "--output-csv", default="artifacts/results/temporal_sweep.csv"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("forwarded", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = (
        args.forwarded[1:] if args.forwarded[:1] == ["--"]
        else args.forwarded
    )
    forbidden = {
        "--temporal-selector", "--token-selector",
        "--speculative-draft-steps", "--event-max-events",
        "--event-min-spacing", "--event-throughput-threshold",
        "--event-buffer-threshold", "--event-bitrate-jump-threshold",
    }
    if forbidden.intersection(forwarded):
        parser.error("temporal options are controlled by the sweep script")
    if any(count <= 0 for count in args.max_events):
        parser.error("--max-events values must be positive")
    if args.min_spacing <= 0:
        parser.error("--min-spacing must be positive")
    if any(value <= 0 for value in (
        args.throughput_threshold, args.buffer_threshold,
        args.bitrate_jump_threshold,
    )):
        parser.error("event thresholds must be positive")

    rows = []
    for selector, count in configurations(args.max_events):
        command = command_for(
            selector, count, args.min_spacing, args.throughput_threshold,
            args.buffer_threshold, args.bitrate_jump_threshold, forwarded,
        )
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(
            selector, count, args.min_spacing, args.throughput_threshold,
            args.buffer_threshold, args.bitrate_jump_threshold,
        )
        with metrics_path.open(encoding="utf-8") as stream:
            row = json.load(stream)
        row["metrics_path"] = str(metrics_path)
        rows.append(row)

    if args.dry_run:
        return
    output_path = ABR_ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "temporal_selector", "event_max_events", "event_min_spacing",
        "event_throughput_threshold", "event_buffer_threshold",
        "event_bitrate_jump_threshold", "mean_reward",
        "inference_latency_mean_ms", "inference_latency_p95_ms",
        "original_tokens_mean", "selected_tokens_mean",
        "token_reduction_ratio", "temporal_history_reduction_ratio",
        "event_timesteps_selected_mean", "time", "metrics_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
