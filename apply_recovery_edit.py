import sys

path = "/root/verl-agent/examples/data_preprocess/generate_extraction_ops_recovery.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

edits = [
    (
        '    parser.add_argument("--procedural", action="store_true")\n    args = parser.parse_args()\n',
        '    parser.add_argument("--procedural", action="store_true")\n'
        '    parser.add_argument("--random-maps", action="store_true")\n    args = parser.parse_args()\n',
    ),
    (
        "        ExtractionOpsWorld(seed=args.seed_start + i, max_steps=args.max_steps, procedural=args.procedural)\n"
        "        for i in range(args.episodes)\n",
        "        ExtractionOpsWorld(\n"
        "            seed=args.seed_start + i, max_steps=args.max_steps,\n"
        "            procedural=args.procedural, random_maps=args.random_maps,\n"
        "        )\n"
        "        for i in range(args.episodes)\n",
    ),
    (
        '        "procedural": args.procedural,\n'
        '        "max_steps": args.max_steps,\n',
        '        "procedural": args.procedural,\n'
        '        "random_maps": args.random_maps,\n'
        '        "max_steps": args.max_steps,\n',
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
print("generate_extraction_ops_recovery.py updated")
