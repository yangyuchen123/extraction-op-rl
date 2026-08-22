import sys

path = "/root/verl-agent/examples/data_preprocess/generate_extraction_ops_recovery.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = (
    "            for local, episode_index in enumerate(indices):\n"
    "                observation = observations[episode_index]\n"
    "                expert_action = experts[episode_index].act(observation)\n"
    '                observation_hash = "sha256:" + hashlib.sha256(observation.encode("utf-8")).hexdigest()\n'
    "                policy_action = policy_actions[local]\n"
    "                agrees = bool(projection_valids[local]) and policy_action == expert_action\n"
    "                disagreement_count += int(not agrees)\n"
    "                if observation_hash not in seen_observations:\n"
    "                    seen_observations.add(observation_hash)\n"
    "                    recovery_rows.append({\n"
    '                        "episode_id": f"recovery-{args.seed_start + episode_index}",\n'
    '                        "seed": args.seed_start + episode_index,\n'
    '                        "step_index": len(policy_traces[episode_index]),\n'
    '                        "observation": observation,\n'
    '                        "observation_hash": observation_hash,\n'
    '                        "action": expert_action,\n'
    '                        "response": f"<action>{expert_action}</action>",\n'
    '                        "policy_response": responses[local],\n'
    '                        "policy_action": policy_action,\n'
    '                        "policy_format_valid": bool(projection_valids[local]),\n'
    '                        "policy_agrees_with_expert": agrees,\n'
    '                        "source": "policy_visited_expert_correction",\n'
    "                    })\n"
    "                else:\n"
    "                    duplicate_observations += 1\n"
)

new = (
    "            for local, episode_index in enumerate(indices):\n"
    "                world = worlds[episode_index]\n"
    "                observation = observations[episode_index]\n"
    "                expert_action = experts[episode_index].act(observation)\n"
    "                # Dedup on semantic state (seed + location + inventory +\n"
    "                # milestones), not raw text: raw text embeds the clock so a\n"
    "                # room loop never collapses.\n"
    "                semantic_key = json.dumps(\n"
    "                    [\n"
    "                        args.seed_start + episode_index,\n"
    "                        world.location,\n"
    "                        sorted(world._inventory()),\n"
    "                        sorted(world.milestones),\n"
    "                    ],\n"
    "                    sort_keys=True,\n"
    "                )\n"
    '                observation_hash = "sha256:" + hashlib.sha256(semantic_key.encode("utf-8")).hexdigest()\n'
    "                policy_action = policy_actions[local]\n"
    "                agrees = bool(projection_valids[local]) and policy_action == expert_action\n"
    "                disagreement_count += int(not agrees)\n"
    "                if observation_hash not in seen_observations:\n"
    "                    seen_observations.add(observation_hash)\n"
    "                    recovery_rows.append({\n"
    '                        "episode_id": f"recovery-{args.seed_start + episode_index}",\n'
    '                        "seed": args.seed_start + episode_index,\n'
    '                        "step_index": len(policy_traces[episode_index]),\n'
    '                        "observation": observation,\n'
    '                        "observation_hash": observation_hash,\n'
    '                        "location": world.location,\n'
    '                        "inventory": sorted(world._inventory()),\n'
    '                        "milestones": sorted(world.milestones),\n'
    '                        "action": expert_action,\n'
    '                        "response": f"<action>{expert_action}</action>",\n'
    '                        "policy_response": responses[local],\n'
    '                        "policy_action": policy_action,\n'
    '                        "policy_format_valid": bool(projection_valids[local]),\n'
    '                        "policy_agrees_with_expert": agrees,\n'
    '                        "source": "policy_visited_expert_correction",\n'
    "                    })\n"
    "                else:\n"
    "                    duplicate_observations += 1\n"
)

count = src.count(old)
if count != 1:
    print(f"ERROR: expected 1 match, found {count}")
    sys.exit(1)
src = src.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("semantic dedup fix applied")
