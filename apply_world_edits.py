import sys

path = "/root/verl-agent/agent_system/environments/env_package/extraction_ops/world.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

edits = [
    (
        "from dataclasses import dataclass\nfrom typing import Any\n",
        "from dataclasses import dataclass\nfrom typing import Any\n\nfrom .generator import generate_random_definition\n",
    ),
    (
        "    def __init__(\n"
        "        self,\n"
        "        seed: int = 20260714,\n"
        "        max_steps: int = 60,\n"
        "        deadline_seconds: int = 35 * 60,\n"
        "        procedural: bool = False,\n"
        "    ):\n"
        "        self.default_seed = seed\n"
        "        self.max_steps = max_steps\n"
        "        self.deadline_seconds = deadline_seconds\n"
        "        self.procedural = procedural\n"
        "        self.reset(seed)\n",
        "    def __init__(\n"
        "        self,\n"
        "        seed: int = 20260714,\n"
        "        max_steps: int = 60,\n"
        "        deadline_seconds: int = 35 * 60,\n"
        "        procedural: bool = False,\n"
        "        random_maps: bool = False,\n"
        "        random_maps_config: dict[str, Any] | None = None,\n"
        "        definition: dict[str, Any] | None = None,\n"
        "    ):\n"
        "        self.default_seed = seed\n"
        "        self.max_steps = max_steps\n"
        "        self.deadline_seconds = deadline_seconds\n"
        "        self.procedural = procedural\n"
        "        self.random_maps = random_maps\n"
        "        self.random_maps_config = random_maps_config\n"
        "        self._definition_override = copy.deepcopy(definition) if definition is not None else None\n"
        "        self.reset(seed)\n",
    ),
    (
        "        rng = random.Random(self.seed)\n"
        "        definition = _procedural_definition(self.seed) if self.procedural else _fixed_definition()\n"
        "        self.variant_id = definition[\"variant_id\"]\n"
        "        self.regions = definition[\"regions\"]\n",
        "        rng = random.Random(self.seed)\n"
        "        if self._definition_override is not None:\n"
        "            definition = self._definition_override\n"
        "        elif self.random_maps:\n"
        "            definition = generate_random_definition(self.seed, self.random_maps_config)\n"
        "        elif self.procedural:\n"
        "            definition = _procedural_definition(self.seed)\n"
        "        else:\n"
        "            definition = _fixed_definition()\n"
        "        self.variant_id = definition[\"variant_id\"]\n"
        "        self.layout_id = definition.get(\"layout_id\")\n"
        "        self.regions = definition[\"regions\"]\n",
    ),
    (
        "        self.extraction_points = definition[\"extractions\"]\n"
        "        self.location = \"west_woods\"\n",
        "        self.extraction_points = definition[\"extractions\"]\n"
        "        self.spawn = definition.get(\"spawn\", \"west_woods\")\n"
        "        self.location = self.spawn\n",
    ),
    (
        "        return json.dumps({\n"
        "            \"seed\": self.seed,\n"
        "            \"procedural\": self.procedural,\n"
        "            \"variant_id\": self.variant_id,\n"
        "            \"regions\": self.regions,\n"
        "            \"edges\": self.edges,\n"
        "            \"location\": self.location,\n",
        "        return json.dumps({\n"
        "            \"seed\": self.seed,\n"
        "            \"procedural\": self.procedural,\n"
        "            \"random_maps\": self.random_maps,\n"
        "            \"variant_id\": self.variant_id,\n"
        "            \"layout_id\": self.layout_id,\n"
        "            \"spawn\": self.spawn,\n"
        "            \"regions\": self.regions,\n"
        "            \"edges\": self.edges,\n"
        "            \"location\": self.location,\n",
    ),
]

for old, new in edits:
    count = src.count(old)
    if count != 1:
        print(f"ERROR: expected exactly 1 match, found {count} for:\n{old[:80]}...")
        sys.exit(1)
    src = src.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("world.py updated successfully")
