import sys

path = "/root/verl-agent/agent_system/environments/env_package/extraction_ops/world.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

edits = [
    # 1. Add REWARD_SCHEMES after ACTION_SECONDS.
    (
        'ACTION_SECONDS = {"look": 5, "inventory": 5, "open": 8, "unlock": 12, "search": 25, "pickup": 5, "extract": 20}\n',
        'ACTION_SECONDS = {"look": 5, "inventory": 5, "open": 8, "unlock": 12, "search": 25, "pickup": 5, "extract": 20}\n'
        "\n"
        "# Reward schemes for ablation. ``milestone`` is the legacy default;\n"
        "# ``outcome`` is the Stage-B1 result-oriented scheme (dominant success\n"
        "# signal + weak navigation hints); ``sparse`` is the pure-outcome extreme\n"
        "# used for teaching and ablations. invalid_action stays non-zero in every\n"
        "# scheme because it is a legality signal, not a progress signal.\n"
        "REWARD_SCHEMES = {\n"
        '    "sparse": {\n'
        '        "prerequisite_acquired": 0.0,\n'
        '        "objective_acquired": 0.0,\n'
        '        "mission_success": 1.0,\n'
        '        "extracted_without_objective": 0.0,\n'
        '        "invalid_action": -0.02,\n'
        "    },\n"
        '    "milestone": {\n'
        '        "prerequisite_acquired": 0.05,\n'
        '        "objective_acquired": 0.15,\n'
        '        "mission_success": 1.0,\n'
        '        "extracted_without_objective": 0.05,\n'
        '        "invalid_action": -0.02,\n'
        "    },\n"
        '    "outcome": {\n'
        '        "prerequisite_acquired": 0.02,\n'
        '        "objective_acquired": 0.03,\n'
        '        "mission_success": 1.0,\n'
        '        "extracted_without_objective": 0.0,\n'
        '        "invalid_action": -0.02,\n'
        "    },\n"
        "}\n",
    ),
    # 2. __init__ signature + reward_weights.
    (
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
        "        random_maps: bool = False,\n"
        "        random_maps_config: dict[str, Any] | None = None,\n"
        "        definition: dict[str, Any] | None = None,\n"
        "        reward_scheme: str = \"milestone\",\n"
        "    ):\n"
        "        self.default_seed = seed\n"
        "        self.max_steps = max_steps\n"
        "        self.deadline_seconds = deadline_seconds\n"
        "        self.procedural = procedural\n"
        "        self.random_maps = random_maps\n"
        "        self.random_maps_config = random_maps_config\n"
        "        self.reward_scheme = reward_scheme\n"
        "        self.reward_weights = dict(REWARD_SCHEMES[reward_scheme])\n"
        "        self._definition_override = copy.deepcopy(definition) if definition is not None else None\n"
        "        self.reset(seed)\n",
    ),
    # 3. _invalid reward.
    (
        "    def _invalid(self, reason: str) -> StepResult:\n"
        "        self._finish_step(1)\n"
        "        return StepResult(\n"
        '            self._observation(event=f"failed: {reason}"), -0.02,\n'
        '            self.done and self.terminal_reason not in {"max_steps", "deadline"},\n'
        '            self.terminal_reason in {"max_steps", "deadline"},\n'
        '            self._info(False, error=reason, reward_components={"environment_invalid_action": -0.02}),\n'
        "        )\n",
        "    def _invalid(self, reason: str) -> StepResult:\n"
        "        self._finish_step(1)\n"
        '        penalty = self.reward_weights["invalid_action"]\n'
        "        return StepResult(\n"
        '            self._observation(event=f"failed: {reason}"), penalty,\n'
        '            self.done and self.terminal_reason not in {"max_steps", "deadline"},\n'
        '            self.terminal_reason in {"max_steps", "deadline"},\n'
        '            self._info(False, error=reason, reward_components={"environment_invalid_action": penalty}),\n'
        "        )\n",
    ),
    # 4. pickup milestones.
    (
        "            if item_id == self.mission[\"prerequisite_item\"] and \"prerequisite_acquired\" not in self.milestones:\n"
        '                self.milestones.add("prerequisite_acquired")\n'
        "                reward += 0.05\n"
        '                reward_components["prerequisite_acquired"] = 0.05\n'
        "            if item[\"mission\"] and \"objective_acquired\" not in self.milestones:\n"
        '                self.milestones.add("objective_acquired")\n'
        "                reward += 0.15\n"
        '                reward_components["objective_acquired"] = 0.15\n',
        "            if item_id == self.mission[\"prerequisite_item\"] and \"prerequisite_acquired\" not in self.milestones:\n"
        '                self.milestones.add("prerequisite_acquired")\n'
        '                reward += self.reward_weights["prerequisite_acquired"]\n'
        '                reward_components["prerequisite_acquired"] = self.reward_weights["prerequisite_acquired"]\n'
        "            if item[\"mission\"] and \"objective_acquired\" not in self.milestones:\n"
        '                self.milestones.add("objective_acquired")\n'
        '                reward += self.reward_weights["objective_acquired"]\n'
        '                reward_components["objective_acquired"] = self.reward_weights["objective_acquired"]\n',
    ),
    # 5. extract rewards.
    (
        "            if has_objective:\n"
        "                self.won = True\n"
        '                self.terminal_reason = "mission_success"\n'
        "                reward += 1.0\n"
        '                reward_components["mission_success"] = 1.0\n'
        '                event = "mission objective carried; extraction succeeded"\n'
        "            else:\n"
        '                self.terminal_reason = "extracted_without_objective"\n'
        "                reward += 0.05\n"
        '                reward_components["extracted_without_objective"] = 0.05\n'
        '                event = "extracted without the mission objective"\n',
        "            if has_objective:\n"
        "                self.won = True\n"
        '                self.terminal_reason = "mission_success"\n'
        '                reward += self.reward_weights["mission_success"]\n'
        '                reward_components["mission_success"] = self.reward_weights["mission_success"]\n'
        '                event = "mission objective carried; extraction succeeded"\n'
        "            else:\n"
        '                self.terminal_reason = "extracted_without_objective"\n'
        '                reward += self.reward_weights["extracted_without_objective"]\n'
        '                reward_components["extracted_without_objective"] = self.reward_weights["extracted_without_objective"]\n'
        '                event = "extracted without the mission objective"\n',
    ),
]

for old, new in edits:
    count = src.count(old)
    if count != 1:
        print(f"ERROR: expected exactly 1 match, found {count} for:\n{old[:100]}...")
        sys.exit(1)
    src = src.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("world.py reward schemes updated successfully")
