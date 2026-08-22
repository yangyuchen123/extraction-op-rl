import sys

edits_by_file = {
    "/root/verl-agent/agent_system/environments/env_package/extraction_ops/envs.py": [
        (
            "    def __init__(self, seeds: list[int], max_steps: int, procedural: bool, deadline_seconds: int):\n"
            "        self.seeds = [int(seed) for seed in seeds]\n"
            "        self.worlds = [\n"
            "            ExtractionOpsWorld(\n"
            "                seed=seed,\n"
            "                max_steps=max_steps,\n"
            "                procedural=procedural,\n"
            "                deadline_seconds=deadline_seconds,\n"
            "            )\n"
            "            for seed in self.seeds\n"
            "        ]\n",
            "    def __init__(\n"
            "        self,\n"
            "        seeds: list[int],\n"
            "        max_steps: int,\n"
            "        procedural: bool,\n"
            "        deadline_seconds: int,\n"
            "        reward_scheme: str,\n"
            "        random_maps: bool,\n"
            "        random_maps_config: dict | None,\n"
            "    ):\n"
            "        self.seeds = [int(seed) for seed in seeds]\n"
            "        self.worlds = [\n"
            "            ExtractionOpsWorld(\n"
            "                seed=seed,\n"
            "                max_steps=max_steps,\n"
            "                procedural=procedural,\n"
            "                deadline_seconds=deadline_seconds,\n"
            "                reward_scheme=reward_scheme,\n"
            "                random_maps=random_maps,\n"
            "                random_maps_config=random_maps_config,\n"
            "            )\n"
            "            for seed in self.seeds\n"
            "        ]\n",
        ),
        (
            "        max_steps = int(env_kwargs.get(\"max_steps\", 60))\n"
            "        procedural = bool(env_kwargs.get(\"procedural\", False))\n"
            "        deadline_seconds = int(env_kwargs.get(\"deadline_seconds\", 35 * 60))\n"
            "        envs_per_worker = max(1, int(env_kwargs.get(\"envs_per_worker\", 1)))\n",
            "        max_steps = int(env_kwargs.get(\"max_steps\", 60))\n"
            "        procedural = bool(env_kwargs.get(\"procedural\", False))\n"
            "        reward_scheme = str(env_kwargs.get(\"reward_scheme\", \"milestone\"))\n"
            "        random_maps = bool(env_kwargs.get(\"random_maps\", False))\n"
            "        random_maps_config_raw = env_kwargs.get(\"random_maps_config\")\n"
            "        random_maps_config = dict(random_maps_config_raw) if random_maps_config_raw else None\n"
            "        deadline_seconds = int(env_kwargs.get(\"deadline_seconds\", 35 * 60))\n"
            "        envs_per_worker = max(1, int(env_kwargs.get(\"envs_per_worker\", 1)))\n",
        ),
        (
            "        self.workers = [\n"
            "            worker_cls.remote(shard, max_steps, procedural, deadline_seconds)\n"
            "            for shard in self.seed_shards\n"
            "        ]\n",
            "        self.workers = [\n"
            "            worker_cls.remote(\n"
            "                shard, max_steps, procedural, deadline_seconds,\n"
            "                reward_scheme, random_maps, random_maps_config,\n"
            "            )\n"
            "            for shard in self.seed_shards\n"
            "        ]\n",
        ),
    ],
    "/root/verl-agent/agent_system/environments/env_manager.py": [
        (
            "        extraction_kwargs = {\n"
            '            "max_steps": config.env.max_steps,\n'
            '            "procedural": config.env.get("procedural", False),\n'
            '            "deadline_seconds": config.env.get("deadline_seconds", 35 * 60),\n'
            '            "envs_per_worker": config.env.get("envs_per_worker", 1),\n'
            "        }\n",
            "        extraction_kwargs = {\n"
            '            "max_steps": config.env.max_steps,\n'
            '            "procedural": config.env.get("procedural", False),\n'
            '            "reward_scheme": config.env.get("reward_scheme", "milestone"),\n'
            '            "random_maps": config.env.get("random_maps", False),\n'
            '            "random_maps_config": config.env.get("random_maps_config"),\n'
            '            "deadline_seconds": config.env.get("deadline_seconds", 35 * 60),\n'
            '            "envs_per_worker": config.env.get("envs_per_worker", 1),\n'
            "        }\n",
        ),
    ],
    "/root/verl-agent/examples/data_preprocess/generate_extraction_ops_expert.py": [
        (
            "def run_episode(\n"
            "    seed: int,\n"
            "    max_steps: int = 60,\n"
            "    procedural: bool = False,\n"
            ") -> tuple[dict[str, Any], list[dict[str, Any]]]:\n"
            "    world = ExtractionOpsWorld(seed=seed, max_steps=max_steps, procedural=procedural)\n",
            "def run_episode(\n"
            "    seed: int,\n"
            "    max_steps: int = 60,\n"
            "    procedural: bool = False,\n"
            "    random_maps: bool = False,\n"
            ") -> tuple[dict[str, Any], list[dict[str, Any]]]:\n"
            "    world = ExtractionOpsWorld(\n"
            "        seed=seed, max_steps=max_steps, procedural=procedural, random_maps=random_maps\n"
            "    )\n",
        ),
        (
            '        "variant_id": world.variant_id,\n'
            '        "procedural": procedural,\n'
            '        "final_snapshot_hash": hashlib.sha256(world.snapshot().encode("utf-8")).hexdigest(),\n',
            '        "variant_id": world.variant_id,\n'
            '        "procedural": procedural,\n'
            '        "random_maps": random_maps,\n'
            '        "layout_id": world.layout_id,\n'
            '        "final_snapshot_hash": hashlib.sha256(world.snapshot().encode("utf-8")).hexdigest(),\n',
        ),
        (
            '    parser.add_argument("--procedural", action="store_true")\n',
            '    parser.add_argument("--procedural", action="store_true")\n'
            '    parser.add_argument("--random-maps", action="store_true")\n',
        ),
        (
            "        summary, episode_steps = run_episode(\n"
            "            args.seed_start + offset,\n"
            "            args.max_steps,\n"
            "            procedural=args.procedural,\n"
            "        )\n",
            "        summary, episode_steps = run_episode(\n"
            "            args.seed_start + offset,\n"
            "            args.max_steps,\n"
            "            procedural=args.procedural,\n"
            "            random_maps=args.random_maps,\n"
            "        )\n",
        ),
        (
            '        "max_steps": args.max_steps,\n'
            '        "procedural": args.procedural,\n'
            '        "variant_counts": dict(sorted(Counter(str(row["variant_id"]) for row in summaries).items())),\n',
            '        "max_steps": args.max_steps,\n'
            '        "procedural": args.procedural,\n'
            '        "random_maps": args.random_maps,\n'
            '        "variant_counts": dict(sorted(Counter(str(row["variant_id"]) for row in summaries).items())),\n',
        ),
    ],
}

for path, edits in edits_by_file.items():
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new in edits:
        count = src.count(old)
        if count != 1:
            print(f"ERROR in {path}: expected exactly 1 match, found {count} for:\n{old[:120]}...")
            sys.exit(1)
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"OK: {path}")

print("all wiring + generate script edits applied")
