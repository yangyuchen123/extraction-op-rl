"""Three-layer validation for Extraction Ops RL analysis.

Layers (each independently summarized):
- fixed_val:   64 fixed held-out seeds (same maps for every checkpoint)
               -> "is the model really getting better on a fixed benchmark?"
- rolling_val: 64 fresh seeds per checkpoint (offset rotates per step)
               -> "is it generalizing, or just memorizing fixed maps?"
- train_like:  64 seeds from the training distribution (train env seed base)
               -> "is it learning the training task?"

Loading supports either a merged model (--model) or a PEFT adapter over a base
model (--base-model + --adapter), the latter for per-step RL LoRA checkpoints.

Usage:
  python evaluate_extraction_ops_three_layer.py \
    --base-model /root/autodl-tmp/models/Qwen3-0.6B-extraction-ops-random-sft \
    --adapter /path/to/global_step_5/actor/lora_adapter \
    --episodes-per-layer 64 \
    --fixed-seed-base 5000 \
    --rolling-seed-base 6000 --rolling-offset 5 \
    --train-seed-base 20260714 \
    --output eval_step5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.env_package.extraction_ops import ExtractionOpsWorld
from agent_system.environments.env_package.extraction_ops.projection import extraction_ops_projection
from agent_system.environments.prompts.extraction_ops import EXTRACTION_OPS_TEMPLATE


def build_worlds(seeds: list[int], max_steps: int, reward_scheme: str) -> list[ExtractionOpsWorld]:
    return [
        ExtractionOpsWorld(seed=seed, max_steps=max_steps, random_maps=True, reward_scheme=reward_scheme)
        for seed in seeds
    ]


def detect_loops(trace: list[dict]) -> dict:
    """Detect repeated (location, inventory, milestones) states in a trace.

    This key intentionally excludes step_count/world_time so a harmful
    A->B->A->B cycle is detected even though the clock advances every step.
    """
    from collections import defaultdict
    positions: dict[tuple, list[int]] = defaultdict(list)
    for step in trace:
        key = (
            step["location"],
            tuple(step.get("inventory", [])),
            tuple(step.get("milestones", [])),
        )
        positions[key].append(step["step"])
    repeated = {k: v for k, v in positions.items() if len(v) > 1}
    if not repeated:
        return {"loop_detected": False, "repeated_state_count": 0, "max_repeat_count": 1}
    max_key = max(repeated, key=lambda k: len(repeated[k]))
    return {
        "loop_detected": True,
        "repeated_state_count": len(repeated),
        "max_repeat_count": len(repeated[max_key]),
        "most_repeated": {
            "location": max_key[0],
            "inventory": list(max_key[1]),
            "milestones": list(max_key[2]),
            "occurrences": repeated[max_key],
        },
    }


def evaluate_seeds(model, tokenizer, seeds, max_steps, max_new_tokens, batch_size, reward_scheme) -> dict:
    worlds = build_worlds(seeds, max_steps, reward_scheme)
    observations = [world.reset(seed)[0] for world, seed in zip(worlds, seeds)]
    rewards = [0.0] * len(seeds)
    format_valid = [0] * len(seeds)
    environment_valid = [0] * len(seeds)
    decisions = [0] * len(seeds)
    traces: list[list[dict]] = [[] for _ in worlds]

    while not all(world.done for world in worlds):
        active = [i for i, world in enumerate(worlds) if not world.done]
        for start in range(0, len(active), batch_size):
            indices = active[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": EXTRACTION_OPS_TEMPLATE.format(current_observation=observations[i])}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for i in indices
            ]
            batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            suffixes = generated[:, batch["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(suffixes, skip_special_tokens=True)
            actions, valids = extraction_ops_projection(responses)
            for local, episode_index in enumerate(indices):
                world = worlds[episode_index]
                result = world.step(actions[local])
                decisions[episode_index] += 1
                format_valid[episode_index] += int(bool(valids[local]))
                environment_valid[episode_index] += int(bool(valids[local]) and result.info["environment_action_valid"])
                rewards[episode_index] += result.reward
                traces[episode_index].append({
                    "step": decisions[episode_index] - 1,
                    "action": actions[local],
                    "format_valid": bool(valids[local]),
                    "environment_action_valid": result.info["environment_action_valid"],
                    "reward": result.reward,
                    "location": result.info["location"],
                    "inventory": list(result.info["inventory"]),
                    "milestones": sorted(world.milestones),
                    "state_hash": result.info["state_hash"],
                    "reward_components": result.info.get("reward_components"),
                    "world_time_seconds": result.info["world_time_seconds"],
                    "terminal_reason": result.info["terminal_reason"],
                })
                observations[episode_index] = result.observation

    successes = sum(int(world.won) for world in worlds)
    total_decisions = sum(decisions)
    loop_stats = [detect_loops(trace) for trace in traces]
    loop_episodes = sum(1 for s in loop_stats if s["loop_detected"])
    summary = {
        "episode_count": len(seeds),
        "success_count": successes,
        "success_rate": successes / max(len(seeds), 1),
        "mean_reward": sum(rewards) / max(len(seeds), 1),
        "mean_episode_length": total_decisions / max(len(seeds), 1),
        "format_valid_ratio": sum(format_valid) / max(total_decisions, 1),
        "environment_action_valid_ratio": sum(environment_valid) / max(total_decisions, 1),
        "terminal_reasons": dict(sorted(Counter(world.terminal_reason for world in worlds).items())),
        "loop_episode_count": loop_episodes,
        "loop_episode_ratio": loop_episodes / max(len(seeds), 1),
        "mean_max_repeat_count": sum(s["max_repeat_count"] for s in loop_stats) / max(len(seeds), 1),
    }
    return {
        "summary": summary,
        "episodes": [
            {
                "seed": seeds[i],
                "success": worlds[i].won,
                "terminal_reason": worlds[i].terminal_reason,
                "reward": rewards[i],
                "length": decisions[i],
                "loop_diagnostics": loop_stats[i],
                "trace": traces[i],
            }
            for i in range(len(seeds))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None, help="Merged model path (alternative to base+adapter)")
    parser.add_argument("--base-model", type=Path, default=None, help="Base model for PEFT adapter loading")
    parser.add_argument("--adapter", type=Path, default=None, help="PEFT adapter dir (e.g. .../actor/lora_adapter)")
    parser.add_argument("--episodes-per-layer", type=int, default=64)
    parser.add_argument("--fixed-seed-base", type=int, default=5000)
    parser.add_argument("--rolling-seed-base", type=int, default=6000)
    parser.add_argument("--rolling-offset", type=int, default=0, help="Checkpoint step index; rolling seeds = base + offset*N")
    parser.add_argument("--train-seed-base", type=int, default=20260714)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--reward-scheme",
        type=str,
        default="outcome",
        help="Must match the training-side reward scheme. Default outcome (premature "
        "extraction = 0.0). WARNING: milestone gives +0.05 to premature extraction "
        "and silently rewards shortcut behavior (the 2026-08-22 reward-trap finding).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch  # noqa: F401  (already imported at module level)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.adapter is not None:
        if args.base_model is None:
            raise SystemExit("--base-model is required when --adapter is provided")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, args.adapter)
        model = model.merge_and_unload()
        tokenizer_path = args.base_model
    elif args.model is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
        )
        tokenizer_path = args.model
    else:
        raise SystemExit("provide --model OR (--base-model + --adapter)")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")

    n = args.episodes_per_layer
    layers = {
        "fixed_val": list(range(args.fixed_seed_base, args.fixed_seed_base + n)),
        "rolling_val": list(range(args.rolling_seed_base + args.rolling_offset * n,
                                   args.rolling_seed_base + (args.rolling_offset + 1) * n)),
        "train_like": list(range(args.train_seed_base, args.train_seed_base + n)),
    }

    payload = {
        "model": str(args.model or args.adapter),
        "base_model": str(args.base_model) if args.base_model else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "rolling_offset": args.rolling_offset,
        "episodes_per_layer": n,
        "reward_scheme": args.reward_scheme,
        "layers": {},
    }
    for layer_name, seeds in layers.items():
        print(f"evaluating {layer_name} ({len(seeds)} seeds)...", file=sys.stderr)
        payload["layers"][layer_name] = evaluate_seeds(
            model, tokenizer, seeds, args.max_steps, args.max_new_tokens, args.batch_size, args.reward_scheme
        )
        summary = payload["layers"][layer_name]["summary"]
        print(f"  {layer_name}: success={summary['success_rate']:.3f} "
              f"mean_len={summary['mean_episode_length']:.1f}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
