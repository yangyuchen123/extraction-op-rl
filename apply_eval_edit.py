import sys

path = "/root/verl-agent/examples/evaluation/evaluate_extraction_ops_policy.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

edits = [
    (
        '    parser.add_argument("--procedural", action="store_true")\n    args = parser.parse_args()\n',
        '    parser.add_argument("--procedural", action="store_true")\n'
        '    parser.add_argument("--random-maps", action="store_true")\n    args = parser.parse_args()\n',
    ),
    (
        "    worlds = [\n"
        "        ExtractionOpsWorld(\n"
        "            seed=args.seed_start + i,\n"
        "            max_steps=args.max_steps,\n"
        "            procedural=args.procedural,\n"
        "        )\n"
        "        for i in range(args.episodes)\n"
        "    ]\n",
        "    worlds = [\n"
        "        ExtractionOpsWorld(\n"
        "            seed=args.seed_start + i,\n"
        "            max_steps=args.max_steps,\n"
        "            procedural=args.procedural,\n"
        "            random_maps=args.random_maps,\n"
        "        )\n"
        "        for i in range(args.episodes)\n"
        "    ]\n",
    ),
    (
        '        "max_new_tokens": args.max_new_tokens,\n'
        '        "procedural": args.procedural,\n'
        '        "variant_counts": dict(sorted(Counter(str(world.variant_id) for world in worlds).items())),\n',
        '        "max_new_tokens": args.max_new_tokens,\n'
        '        "procedural": args.procedural,\n'
        '        "random_maps": args.random_maps,\n'
        '        "variant_counts": dict(sorted(Counter(str(world.variant_id) for world in worlds).items())),\n',
    ),
]

for old, new in edits:
    count = src.count(old)
    if count != 1:
        print(f"ERROR: expected 1 match, found {count} for:\n{old[:100]}...")
        sys.exit(1)
    src = src.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("evaluate_extraction_ops_policy.py updated")
