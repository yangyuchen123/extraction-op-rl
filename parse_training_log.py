"""Parse a verl main_ppo training log into structured per-step metric JSON.

Extracts every ``step:N - key:value ...`` line and keeps the metrics relevant
for diagnostics (success_rate, advantages, grad_norm, reward, length, etc.).

Usage:
  python parse_training_log.py <training.log> <output.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys

KEEP_PREFIXES = (
    "val/success_rate",
    "val/extraction_ops/test_score",
    "episode/success_rate",
    "episode/reward",
    "episode/length",
    "critic/score",
    "critic/rewards",
    "critic/advantages",
    "critic/returns",
    "actor/grad_norm",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/kl_loss",
    "actor/entropy_loss",
    "response_length",
    "perf/throughput",
    "perf/max_memory_allocated_gb",
    "timing_s/step",
    "training/global_step",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    steps = []
    pattern = re.compile(r"(?:^|\s)([a-zA-Z0-9_/.-]+):(-?[0-9.]+)")
    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "step:" not in line or " - " not in line:
                continue
            if "val/" not in line and "episode/" not in line and "actor/" not in line and "critic/" not in line:
                continue
            # Extract the step number at the start of the metric block.
            step_match = re.search(r"step:(\d+)\s+-", line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            metrics = {}
            for key, val in pattern.findall(line):
                if key.startswith(KEEP_PREFIXES):
                    metrics[key] = float(val)
            if metrics:
                metrics["step"] = step
                steps.append(metrics)

    # Deduplicate by step (logs can repeat lines via tqdm carriage returns).
    by_step = {}
    for m in steps:
        by_step[m["step"]] = m
    ordered = [by_step[k] for k in sorted(by_step)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(ordered, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\n-> {len(ordered)} steps saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
