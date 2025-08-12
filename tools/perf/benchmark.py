#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from statistics import median


def timed_operation(iterations: int) -> float:
    start = time.perf_counter()
    # Placeholder: CPU-bound loop to simulate work; replace with real perf target
    total = 0
    for i in range(iterations):
        total += (i * i) % 97
    _ = total  # avoid optimization
    end = time.perf_counter()
    return end - start


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--baseline", type=str, required=False)
    parser.add_argument("--save", type=str, required=False)
    parser.add_argument("--regression-threshold", type=float, default=5.0, help="Allowed slowdown percentage before fail")
    args = parser.parse_args()

    durations = [timed_operation(args.iterations) for _ in range(args.runs)]
    current = {
        "iterations": args.iterations,
        "runs": args.runs,
        "durations_sec": durations,
        "median_sec": median(durations),
    }

    if args.save:
        save_json(args.save, current)

    if args.baseline and os.path.exists(args.baseline):
        baseline = load_json(args.baseline)
        base_median = baseline.get("median_sec")
        if not base_median:
            print("Baseline missing median_sec; skipping regression check", file=sys.stderr)
            return 0
        curr_median = current["median_sec"]
        slowdown_pct = ((curr_median - base_median) / base_median) * 100.0
        print(f"Baseline median: {base_median:.6f}s, Current median: {curr_median:.6f}s, Slowdown: {slowdown_pct:.2f}%")
        if slowdown_pct > args.regression_threshold:
            print(f"Performance regression detected (> {args.regression_threshold}% slowdown)", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())