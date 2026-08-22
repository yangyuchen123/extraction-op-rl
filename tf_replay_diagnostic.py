"""B1.5-2: teacher-forcing replay diagnostic.

Answers: does the trained model reproduce expert actions when given the
*exact training prompts* (greedy decode, same settings as evaluation)?

- replay accuracy high (>90%) but online rollout poor
  -> exposure bias / frequency bias (training-time distribution shift)
- replay accuracy also low
  -> prompt/data inconsistency bug (train prompt != eval prompt)

Usage:
  python tf_replay_diagnostic.py \
    --base-model /root/autodl-tmp/models/Qwen3-0.6B \
    --adapter  /root/autodl-tmp/checkpoints/.../global_step_XXX \
    --parquet  /root/data/verl-agent/extraction_ops_random_sft/train.parquet \
    --n-samples 300
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")

    df = pd.read_parquet(args.parquet)
    print(f"parquet rows: {len(df)}", flush=True)

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(df)), min(args.n_samples, len(df)))
    indices.sort()  # preserve trajectory order for position analysis

    exact = 0
    per_action = Counter()
    per_action_total = Counter()
    detail = []
    for start in range(0, len(indices), args.batch_size):
        idxs = indices[start : start + args.batch_size]
        prompts = [df.iloc[i]["prompt"] for i in idxs]
        golds = [str(df.iloc[i]["response"]).strip() for i in idxs]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        suffixes = generated[:, batch["input_ids"].shape[1] :]
        preds = [t.strip() for t in tokenizer.batch_decode(suffixes, skip_special_tokens=True)]
        for i, (gold, pred) in enumerate(zip(golds, preds)):
            ok = pred == gold
            exact += int(ok)
            action = gold.split()[0] if gold.split() else gold
            per_action[action] += int(ok)
            per_action_total[action] += 1
            detail.append({
                "row": idxs[i],
                "gold": gold,
                "pred": pred[:80],
                "exact": ok,
            })

    acc = exact / len(indices)
    per_action_acc = {a: f"{per_action[a]}/{per_action_total[a]}" for a in per_action_total}
    payload = {
        "n_samples": len(indices),
        "exact_match_accuracy": round(acc, 4),
        "per_action_accuracy": dict(sorted(per_action_acc.items())),
        "head": detail[:10],
        "tail": detail[-10:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"exact-match accuracy: {acc:.3f} ({exact}/{len(indices)})", flush=True)
    print("per-action:", json.dumps(per_action_acc, ensure_ascii=False), flush=True)
    print(f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
