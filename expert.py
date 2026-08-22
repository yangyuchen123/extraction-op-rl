"""Observation-only expert for fixed, procedural, and random Extraction Ops worlds.

The expert reads only public observations: mission intel, the KNOWN MAP brief,
the current EXITS line, and the admissible action list. It never touches
``world.assets``/``world.items`` authoritative state.

The public graph is parsed from the KNOWN MAP brief when present, falling back
to the legacy hard-coded graph so fixed/procedural worlds keep working.
Door handling is generic: when the BFS next-hop is blocked, the expert consults
the EXITS line to find the door on that edge and unlocks/opens it.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class ExpertPolicyError(RuntimeError):
    pass


PUBLIC_GRAPH = {
    "west_woods": {"roadside_cache", "old_road_gate"},
    "roadside_cache": {"west_woods", "dorm_exterior"},
    "dorm_exterior": {"roadside_cache", "dorm_lobby", "east_checkpoint"},
    "dorm_lobby": {"dorm_exterior", "admin_corridor"},
    "admin_corridor": {"dorm_lobby", "room_204", "room_206"},
    "room_204": {"admin_corridor"},
    "room_206": {"admin_corridor"},
    "old_road_gate": {"west_woods"},
    "east_checkpoint": {"dorm_exterior"},
}


def _line_value(observation: str, prefix: str) -> str:
    for line in observation.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise ExpertPolicyError(f"observation is missing {prefix!r}")


def _location(observation: str) -> str:
    return _line_value(observation, "LOCATION:").split(" ", 1)[0]


def _inventory(observation: str) -> set[str]:
    raw = _line_value(observation, "INVENTORY:").split(";", 1)[0].strip()
    return set() if raw == "empty" else {item.strip() for item in raw.split(",") if item.strip()}


def _admissible_actions(observation: str) -> set[str]:
    marker = "ADMISSIBLE ACTIONS:\n"
    if marker not in observation:
        raise ExpertPolicyError("observation is missing admissible actions")
    actions: set[str] = set()
    for line in observation.split(marker, 1)[1].splitlines():
        if line.startswith("TERMINAL:"):
            break
        if line.startswith("- "):
            actions.add(line[2:].strip())
    return actions


def _choose(actions: set[str], *preferences: str) -> str:
    for action in preferences:
        if action in actions:
            return action
    raise ExpertPolicyError(f"expected one of {preferences}; actual={sorted(actions)}")


@dataclass
class ExtractionOpsExpert:
    """Deterministic oracle that uses only mission text and public observations."""

    pace: str = "run"
    mission: dict[str, str] = field(default_factory=dict)
    public_graph: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _exits: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.public_graph = {k: set(v) for k, v in PUBLIC_GRAPH.items()}
        self._exits = {}

    def reset(self) -> None:
        self.mission.clear()
        self.public_graph = {k: set(v) for k, v in PUBLIC_GRAPH.items()}
        self._exits = {}

    def _learn_graph(self, observation: str) -> None:
        if "KNOWN MAP:" not in observation:
            return
        raw = _line_value(observation, "KNOWN MAP:")
        graph: dict[str, set[str]] = {}
        for part in raw.split(";"):
            part = part.strip()
            if "<->" not in part:
                continue
            a, b = [x.strip() for x in part.split("<->", 1)]
            if a and b:
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
        if graph:
            self.public_graph = graph

    def _parse_exits(self, observation: str) -> dict[str, dict[str, Any]]:
        raw = _line_value(observation, "EXITS:").rstrip(".")
        exits: dict[str, dict[str, Any]] = {}
        if raw == "none":
            return exits
        for entry in re.finditer(r"(\S+)\(([^)]*)\)", raw):
            destination = entry.group(1)
            inner = entry.group(2).strip()
            match = re.match(r"(\d+)m, blocked=(True|False)(?:, door=(\S+))?", inner)
            if match:
                exits[destination] = {
                    "blocked": match.group(2) == "True",
                    "door": match.group(3),
                }
        return exits

    def _learn_mission(self, observation: str) -> None:
        if self.mission:
            return
        key = re.fullmatch(r"(\S+) is inside (\S+) at (\S+)\.", _line_value(observation, "KEY INTEL:"))
        target = re.fullmatch(r"(\S+) is inside (\S+) at (\S+); door=(\S+)\.", _line_value(observation, "TARGET INTEL:"))
        extraction = _line_value(observation, "VALID EXTRACTION:").rstrip(".")
        if not key or not target:
            raise ExpertPolicyError("cannot parse mission intel")
        self.mission.update({
            "key_item": key.group(1),
            "key_container": key.group(2),
            "key_location": key.group(3),
            "required_item": target.group(1),
            "target_container": target.group(2),
            "target_location": target.group(3),
            "target_door": target.group(4),
            "extraction": extraction,
        })

    def _shortest_next(self, start: str, goal: str) -> str:
        if start == goal:
            return goal
        queue = deque([(start, [])])
        seen = {start}
        graph = self.public_graph
        while queue:
            node, path = queue.popleft()
            for nxt in sorted(graph.get(node, ())):
                if nxt in seen:
                    continue
                new_path = path + [nxt]
                if nxt == goal:
                    return new_path[0]
                seen.add(nxt)
                queue.append((nxt, new_path))
        raise ExpertPolicyError(f"no public route from {start!r} to {goal!r}")

    def _clear_blocking_door(self, next_location: str, actions: set[str]) -> str | None:
        door = self._exits.get(next_location, {}).get("door")
        if not door:
            return None
        key = self.mission.get("key_item")
        for candidate in (f"unlock {door} with {key}", f"open {door}"):
            if candidate in actions:
                return candidate
        return None

    def act(self, observation: str) -> str:
        if "TERMINAL:" in observation:
            raise ExpertPolicyError("cannot act on a terminal observation")
        self._learn_graph(observation)
        self._learn_mission(observation)
        self._exits = self._parse_exits(observation)
        location = _location(observation)
        inventory = _inventory(observation)
        actions = _admissible_actions(observation)
        if self.mission["required_item"] in inventory:
            return self._go_or_extract(location, actions)
        if self.mission["key_item"] not in inventory:
            return self._go_or_interact(
                location, actions,
                goal=self.mission["key_location"],
                item=self.mission["key_item"],
                container=self.mission["key_container"],
            )
        return self._go_or_interact(
            location, actions,
            goal=self.mission["target_location"],
            item=self.mission["required_item"],
            container=self.mission["target_container"],
        )

    def _go_or_interact(
        self,
        location: str,
        actions: set[str],
        *,
        goal: str,
        item: str,
        container: str,
    ) -> str:
        if location == goal:
            return _choose(actions, f"pickup {item}", f"search {container}", f"open {container}")
        next_location = self._shortest_next(location, goal)
        move = f"move {next_location} {self.pace}"
        if move in actions:
            return move
        clear = self._clear_blocking_door(next_location, actions)
        if clear:
            return clear
        raise ExpertPolicyError(f"route to {goal!r} is blocked; actions={sorted(actions)}")

    def _go_or_extract(self, location: str, actions: set[str]) -> str:
        extraction = self.mission["extraction"]
        if location == extraction:
            return _choose(actions, f"extract {extraction}")
        next_location = self._shortest_next(location, extraction)
        move = f"move {next_location} {self.pace}"
        if move in actions:
            return move
        clear = self._clear_blocking_door(next_location, actions)
        if clear:
            return clear
        raise ExpertPolicyError(f"route to extraction is blocked; actions={sorted(actions)}")
