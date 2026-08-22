"""Teaching demo: dense vs sparse vs outcome reward on a real random map.

Shows, for one map, the per-step reward under three schemes
(sparse / milestone / outcome) for (a) the BFS-expert success trajectory and
(b) a failure trajectory that grabs the objective then loops until max_steps.
"""
from __future__ import annotations

from agent_system.environments.env_package.extraction_ops import ExtractionOpsExpert, ExtractionOpsWorld

SCHEMES = ["sparse", "milestone", "outcome"]


def run_success(seed: int, scheme: str):
    env = ExtractionOpsWorld(seed=seed, random_maps=True, reward_scheme=scheme)
    expert = ExtractionOpsExpert()
    obs, _ = env.reset(seed)
    rows = []
    total = 0.0
    while not env.done:
        action = expert.act(obs)
        result = env.step(action)
        total += result.reward
        rows.append((action, result.reward, result.info.get("reward_components", {})))
        obs = result.observation
    return total, rows, env.terminal_reason


def run_loop(seed: int, scheme: str):
    env = ExtractionOpsWorld(seed=seed, random_maps=True, reward_scheme=scheme)
    expert = ExtractionOpsExpert()
    obs, _ = env.reset(seed)
    total = 0.0
    steps = 0
    got_objective = False
    loop_actions = None
    toggle = False
    while not env.done:
        if not got_objective:
            action = expert.act(obs)
            result = env.step(action)
            total += result.reward
            obs = result.observation
            steps += 1
            if "objective_acquired" in result.info.get("reward_components", {}):
                got_objective = True
                here = env.location
                for a in env.admissible_actions():
                    if a.startswith("move ") and a.endswith(" run"):
                        dest = a.split()[1]
                        if dest != here:
                            loop_actions = [f"move {dest} run", f"move {here} run"]
                            break
                if loop_actions is None:
                    break
        else:
            action = loop_actions[toggle]
            toggle = not toggle
            result = env.step(action)
            total += result.reward
            obs = result.observation
            steps += 1
    return total, env.terminal_reason, steps


def main():
    seed = 42
    print(f"随机地图 seed={seed}\n")

    # Success trajectory: run each scheme, then print a merged table.
    runs = {scheme: run_success(seed, scheme) for scheme in SCHEMES}
    print("=" * 72)
    print("A. 成功轨迹（BFS 专家）—— 每步奖励对比")
    print("=" * 72)
    header = f"{'step':>4} | {'action':<28} | {'sparse':>7} | {'milestone':>9} | {'outcome':>7}"
    print(header)
    print("-" * len(header))
    n = max(len(runs[s][1]) for s in SCHEMES)
    for i in range(n):
        action = runs["milestone"][1][i][0] if i < len(runs["milestone"][1]) else ""
        vals = []
        for s in SCHEMES:
            r = runs[s][1][i][1] if i < len(runs[s][1]) else 0.0
            vals.append(f"{r:+.2f}")
        print(f"{i:>4} | {action:<28} | {vals[0]:>7} | {vals[1]:>9} | {vals[2]:>7}")
    print("-" * len(header))
    totals = " | ".join(f"{runs[s][0]:+.2f}" for s in SCHEMES)
    print(f"{'TOTAL':>4} | {'':<28} | {totals}")
    print()

    # Loop failure: grab objective then loop.
    print("=" * 72)
    print("B. 失败轨迹（拿到账本后循环到 max_steps）—— 总回报对比")
    print("=" * 72)
    for s in SCHEMES:
        total, reason, steps = run_loop(seed, s)
        print(f"  {s:<10} 总回报={total:+.2f}   terminal={reason}  steps={steps}")
    print()
    print("解读：")
    print("  - milestone 给循环轨迹 +0.20（钥匙0.05+账本0.15），模型'拿0.20躺平'")
    print("  - outcome   只给 +0.05（弱信号），成功 +1.0 仍是绝对主导")
    print("  - sparse    循环轨迹 0 分，但代价是成功前梯度全为 0（探索难）")


if __name__ == "__main__":
    main()
