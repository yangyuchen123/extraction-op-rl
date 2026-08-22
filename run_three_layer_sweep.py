"""Run three-layer eval over every saved per-step LoRA adapter, then summarize.

Usage:
  python run_three_layer_sweep.py \
    --adapter-dir /root/autodl-tmp/adapters/b1 \
    --base-model /root/autodl-tmp/models/Qwen3-0.6B-extraction-ops-random-sft \
    --output-dir /root/autodl-tmp/eval_archive/b1 \
    --episodes-per-layer 64
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "evaluate_extraction_ops_three_layer.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-layer", type=int, default=64)
    parser.add_argument("--fixed-seed-base", type=int, default=5000)
    parser.add_argument("--rolling-seed-base", type=int, default=6000)
    parser.add_argument("--train-seed-base", type=int, default=20260714)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapters = sorted(glob.glob(str(args.adapter_dir / "global_step_*")))
    if not adapters:
        raise SystemExit(f"no adapters found under {args.adapter_dir}")

    rows = []
    for adapter in adapters:
        step = int(Path(adapter).name.split("_")[-1])
        out = args.output_dir / f"eval_step{step}.json"
        if out.exists():
            print(f"step {step} already evaluated, skipping", file=sys.stderr)
        else:
            cmd = [
                sys.executable, str(SCRIPT),
                "--base-model", str(args.base_model),
                "--adapter", str(adapter),
                "--episodes-per-layer", str(args.episodes_per_layer),
                "--fixed-seed-base", str(args.fixed_seed_base),
                "--rolling-seed-base", str(args.rolling_seed_base),
                "--rolling-offset", str(step),
                "--train-seed-base", str(args.train_seed_base),
                "--max-steps", str(args.max_steps),
                "--output", str(out),
            ]
            subprocess.run(cmd, check=True)
        data = json.loads(out.read_text(encoding="utf-8"))
        row = {"step": step}
        for layer, payload in data["layers"].items():
            row[layer] = payload["summary"]
        rows.append(row)

    summary_path = args.output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sweep summary -> {summary_path}")

    # Compact learning-curve view.
    print("\n=== learning curves ===")
    header = f"{'step':>5} | {'fixed':>7} | {'rolling':>7} | {'train_like':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        f = row["fixed_val"]["success_rate"]
        r = row["rolling_val"]["success_rate"]
        t = row["train_like"]["success_rate"]
        print(f"{row['step']:>5} | {f:>7.3f} | {r:>7.3f} | {t:>9.3f}")


if __name__ == "__main__":
    main()
