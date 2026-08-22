"""Stage A smoke check for the bounded random map generator.

For every seed:
  1. build the world in ``random_maps`` mode (full pipeline: seed -> generator
     -> world reset);
  2. solve it with the observation-only Expert via BFS;
  3. assert every Expert action is admissible and legal.

Also verifies generator determinism and reports layout diversity. This script
does not touch the legacy fixed/8-variant modes, which are covered by the
regression tests in tests/environments/.

Usage:
  python check_extraction_ops_random_maps.py --num-seeds 200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from agent_system.environments.env_package.extraction_ops import ExtractionOpsExpert, ExtractionOpsWorld
from agent_system.environments.env_package.extraction_ops.generator import (
    generate_random_definition,
    structural_invariants,
)


def solve_one(seed: int, config: dict | None, max_steps: int, deadline_seconds: int) -> dict:
    env = ExtractionOpsWorld(
        seed=seed,
        random_maps=True,
        random_maps_config=config,
        max_steps=max_steps,
        deadline_seconds=deadline_seconds,
    )
    expert = ExtractionOpsExpert()
    observation, _ = env.reset(seed)
    actions: list[str] = []
    total_reward = 0.0
    while not env.done:
        action = expert.act(observation)
        if action not in env.admissible_actions():
            return {
                "ok": False,
                "seed": seed,
                "reason": f"expert_illegal_action={action!r}",
                "steps": len(actions),
                "actions": actions,
            }
        actions.append(action)
        result = env.step(action)
        if not result.info["is_action_valid"]:
            return {
                "ok": False,
                "seed": seed,
                "reason": f"env_invalid_action={action!r}",
                "steps": len(actions),
                "actions": actions,
            }
        total_reward += result.reward
        observation = result.observation
        if len(actions) > max_steps + 5:
            return {
                "ok": False,
                "seed": seed,
                "reason": "exceeded_step_budget",
                "steps": len(actions),
                "actions": actions,
            }
    ok = bool(env.won and env.terminal_reason == "mission_success")
    return {
        "ok": ok,
        "seed": seed,
        "reason": env.terminal_reason,
        "steps": len(actions),
        "reward": total_reward,
        "layout_id": env.layout_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--deadline-seconds", type=int, default=35 * 60)
    parser.add_argument("--config", type=str, default=None, help="JSON config override for the generator")
    args = parser.parse_args()

    config = json.loads(args.config) if args.config else None

    # Determinism check.
    first = generate_random_definition(0, config)
    second = generate_random_definition(0, config)
    if first["layout_id"] != second["layout_id"]:
        print("FAIL: generator is not deterministic for seed 0")
        return 1
    print(f"determinism ok (seed 0 layout_id={first['layout_id']})")

    # Structural invariant check on a sample.
    sample = generate_random_definition(12345, config)
    violations = structural_invariants(sample)
    if violations:
        print("FAIL: structural invariants violated:", violations)
        return 1

    results = []
    layouts: Counter[str] = Counter()
    for seed in range(args.num_seeds):
        result = solve_one(seed, config, args.max_steps, args.deadline_seconds)
        results.append(result)
        if result.get("layout_id"):
            layouts[result["layout_id"]] += 1

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    steps = [r["steps"] for r in results]
    rewards = [r.get("reward", 0.0) for r in results]

    print(f"seeds            = {args.num_seeds}")
    print(f"success          = {len(successes)}/{len(results)}")
    print(f"unique layouts   = {len(layouts)}")
    print(f"steps min/median/max = {min(steps)}/{sorted(steps)[len(steps)//2]}/{max(steps)}")
    print(f"reward mean      = {sum(rewards)/max(len(rewards), 1):.3f}")

    if failures:
        print("\nFAILURES:")
        for r in failures[:20]:
            print(f"  seed={r['seed']} reason={r['reason']} steps={r['steps']}")
        return 1

    print("\nStage A random map smoke: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
