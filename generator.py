"""Bounded random map generator for Extraction Ops.

Produces a random but provably-feasible raid definition:

* a connected "core" graph where spawn, key room, and extraction live;
* exactly one locked target room attached to the core by a single door;
* optional unlocked dead-end rooms attached by doors (open-only obstacles).

The single-leaf locked-room construction guarantees the observation-only
Expert can always solve the episode without privileged world access. That
feasibility invariant is checked by :func:`validate_definition`.

The generator never mutates the legacy fixed/8-variant procedural modes, so
they remain available as regression canaries.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

TERRAINS = ["forest", "road", "interior", "open_ground"]

DEFAULT_RANDOM_CONFIG: dict[str, Any] = {
    "min_core_regions": 5,
    "max_core_regions": 8,
    "min_door_rooms": 0,
    "max_door_rooms": 2,
    # Extra cycles added on top of the core spanning tree.
    "extra_edges": (0, 2),
    "distance_range": (20, 200),  # metres per edge
    # Optional empty containers placed in core rooms (distractors for the model).
    "num_distractor_containers": 0,
}

KEY_ITEM = "room_key"
OBJECTIVE_ITEM = "sealed_ledger"
KEY_CONTAINER = "key_chest"
TARGET_CONTAINER = "target_safe"
TARGET_DOOR = "target_door"


def _spanning_tree(rng: random.Random, nodes: list[str]) -> list[tuple[str, str]]:
    """Return a random spanning tree over ``nodes`` as (a, b) pairs."""
    perm = nodes[:]
    rng.shuffle(perm)
    edges: list[tuple[str, str]] = []
    for i in range(1, len(perm)):
        parent = perm[rng.randrange(i)]
        edges.append((perm[i], parent))
    return edges


def _edge_tuple(rng: random.Random, a: str, b: str, door_id: str | None, cfg: dict[str, Any]):
    return (
        a,
        b,
        rng.randint(*cfg["distance_range"]),
        rng.choice(TERRAINS),
        door_id,
    )


def layout_id(definition: dict[str, Any]) -> str:
    """Stable fingerprint of the map's topology + role assignment.

    Distances, terrain, and display names are excluded, so two maps that only
    differ cosmetically share a layout id. This is the identity used for
    held-out layout deduplication.
    """
    fingerprint = {
        "spawn": definition["spawn"],
        "edges": sorted(
            (tuple(sorted((a, b))), door)
            for a, b, _distance, _terrain, door in definition["edges"]
        ),
        "locked_doors": sorted(
            aid for aid, asset in definition["assets"].items()
            if asset.get("type") == "door" and asset.get("locked")
        ),
        "mission": definition["mission"],
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return "layout:" + digest[:16]


def generate_random_definition(
    seed: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one deterministic, bounded random map definition.

    The returned dict matches the schema produced by ``_fixed_definition`` and
    ``_procedural_definition`` in ``world.py``, plus a ``spawn`` key.
    """
    cfg = dict(DEFAULT_RANDOM_CONFIG)
    if config:
        cfg.update(config)
    rng = random.Random(int(seed))

    num_core = rng.randint(cfg["min_core_regions"], cfg["max_core_regions"])
    num_door_rooms = rng.randint(cfg["min_door_rooms"], cfg["max_door_rooms"])

    core = [f"sector_{i}" for i in range(num_core)]
    door_rooms = [f"sector_{num_core + i}" for i in range(num_door_rooms)]
    target_room = f"sector_{num_core + num_door_rooms}"
    all_rooms = core + door_rooms + [target_room]

    regions = {rid: f"Sector {i}" for i, rid in enumerate(all_rooms)}

    # Connected core: spanning tree + optional extra cycles (no duplicate edges).
    edges: list[tuple[Any, ...]] = []
    for a, b in _spanning_tree(rng, core):
        edges.append(_edge_tuple(rng, a, b, None, cfg))
    existing_pairs = {frozenset((a, b)) for a, b, *_ in edges}
    for _ in range(rng.randint(*cfg["extra_edges"])):
        a, b = rng.sample(core, 2)
        if frozenset((a, b)) in existing_pairs:
            continue
        existing_pairs.add(frozenset((a, b)))
        edges.append(_edge_tuple(rng, a, b, None, cfg))

    # Role assignment inside the core.
    spawn = rng.choice(core)
    target_anchor = rng.choice(core)
    key_room = rng.choice([r for r in core if r != spawn])
    extraction = rng.choice([r for r in core if r not in {spawn, key_room}])

    # Locked target room (single leaf guarded by the only locked door).
    edges.append(_edge_tuple(rng, target_room, target_anchor, TARGET_DOOR, cfg))

    # Optional unlocked dead-end rooms.
    door_anchors: dict[str, str] = {}
    for rid in door_rooms:
        anchor = rng.choice(core)
        door_anchors[rid] = anchor
        edges.append(_edge_tuple(rng, rid, anchor, f"door_{rid}", cfg))

    assets: dict[str, dict[str, Any]] = {
        TARGET_DOOR: {
            "name": "target door",
            "location": target_anchor,
            "type": "door",
            "state": "closed",
            "locked": True,
            "key": KEY_ITEM,
            "searchable": False,
            "searched": False,
        },
    }
    for rid in door_rooms:
        assets[f"door_{rid}"] = {
            "name": f"{rid} door",
            "location": door_anchors[rid],
            "type": "door",
            "state": "closed",
            "locked": False,
            "key": None,
            "searchable": False,
            "searched": False,
        }

    assets[KEY_CONTAINER] = {
        "name": "key chest",
        "location": key_room,
        "type": "container",
        "state": "closed",
        "locked": False,
        "key": None,
        "searchable": True,
        "searched": False,
    }
    assets[TARGET_CONTAINER] = {
        "name": "target safe",
        "location": target_room,
        "type": "container",
        "state": "closed",
        "locked": False,
        "key": None,
        "searchable": True,
        "searched": False,
    }

    for i in range(cfg["num_distractor_containers"]):
        cid = f"distractor_crate_{i}"
        assets[cid] = {
            "name": f"crate {i}",
            "location": rng.choice(core),
            "type": "container",
            "state": "closed",
            "locked": False,
            "key": None,
            "searchable": True,
            "searched": False,
        }

    items: dict[str, dict[str, Any]] = {
        KEY_ITEM: {
            "name": "room key",
            "location": key_room,
            "container": KEY_CONTAINER,
            "slots": 1,
            "weight": 0.1,
            "mission": False,
            "discovered": False,
            "carried": False,
        },
        OBJECTIVE_ITEM: {
            "name": "sealed ledger",
            "location": target_room,
            "container": TARGET_CONTAINER,
            "slots": 2,
            "weight": 1.2,
            "mission": True,
            "discovered": False,
            "carried": False,
        },
    }

    mission = {
        "required_item": OBJECTIVE_ITEM,
        "prerequisite_item": KEY_ITEM,
        "key_location": key_room,
        "key_container": KEY_CONTAINER,
        "target_location": target_room,
        "target_door": TARGET_DOOR,
        "target_container": TARGET_CONTAINER,
        "extraction_point": extraction,
    }

    definition = {
        "variant_id": f"random-{seed}",
        "spawn": spawn,
        "regions": regions,
        "edges": edges,
        "assets": assets,
        "items": items,
        "mission": mission,
        "extractions": [extraction],
    }
    definition["layout_id"] = layout_id(definition)
    return definition


def structural_invariants(definition: dict[str, Any]) -> list[str]:
    """Return a list of violated feasibility invariants (empty means OK).

    These are fast graph checks; an Expert playthrough is the authoritative
    validation in :func:`validate_definition`.
    """
    violations: list[str] = []
    regions = definition["regions"]
    edges = definition["edges"]
    assets = definition["assets"]
    mission = definition["mission"]
    spawn = definition.get("spawn")

    if not regions:
        violations.append("no regions")
    if spawn not in regions:
        violations.append(f"spawn {spawn!r} not in regions")
    for a, b, *_ in edges:
        if a not in regions or b not in regions:
            violations.append(f"edge {a}<->{b} references unknown region")

    # Adjacency including door info, and without locked doors.
    unlocked_adj: dict[str, set[str]] = {r: set() for r in regions}
    full_adj: dict[str, set[str]] = {r: set() for r in regions}
    for a, b, _d, _t, door in edges:
        full_adj[a].add(b)
        full_adj[b].add(a)
        locked = bool(door and assets.get(door, {}).get("locked"))
        if not locked:
            unlocked_adj[a].add(b)
            unlocked_adj[b].add(a)

    def reachable(adj: dict[str, set[str]], start: str) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    if spawn in regions:
        full_reach = reachable(full_adj, spawn)
        if set(regions) - full_reach:
            violations.append("graph is not connected")
        unlocked_reach = reachable(unlocked_adj, spawn)
        if mission["key_location"] not in unlocked_reach:
            violations.append("key location unreachable without a locked door")

    # Mission contract.
    for key in (
        "required_item",
        "prerequisite_item",
        "key_location",
        "key_container",
        "target_location",
        "target_door",
        "target_container",
        "extraction_point",
    ):
        if key not in mission:
            violations.append(f"mission missing {key!r}")

    if mission.get("key_location") == mission.get("target_location"):
        violations.append("key and target share a location")

    target_door = assets.get(mission.get("target_door"))
    if not target_door or target_door.get("type") != "door" or not target_door.get("locked"):
        violations.append("target door missing or not locked")
    elif target_door.get("key") != mission.get("prerequisite_item"):
        violations.append("target door key mismatch")

    for container_id in (mission.get("key_container"), mission.get("target_container")):
        if container_id not in assets:
            violations.append(f"container {container_id!r} missing")

    if mission.get("extraction_point") not in regions:
        violations.append("extraction point not in regions")

    return violations


def validate_definition(
    seed: int,
    definition: dict[str, Any],
    max_steps: int = 60,
    deadline_seconds: int = 35 * 60,
) -> dict[str, Any]:
    """Authoritative feasibility check: structural invariants + Expert playthrough.

    Imports world/expert lazily to keep this module importable in lightweight
    data-generation processes.
    """
    from .expert import ExtractionOpsExpert
    from .world import ExtractionOpsWorld

    violations = structural_invariants(definition)
    if violations:
        return {
            "ok": False,
            "seed": seed,
            "layout_id": definition.get("layout_id"),
            "structural_violations": violations,
            "expert_success": False,
            "terminal_reason": "structural_failure",
            "steps": 0,
            "reward": 0.0,
        }

    env = ExtractionOpsWorld(
        seed=seed,
        max_steps=max_steps,
        deadline_seconds=deadline_seconds,
        definition=copy.deepcopy(definition),
    )
    expert = ExtractionOpsExpert()
    observation, _ = env.reset(seed)
    total_reward = 0.0
    steps = 0
    while not env.done:
        if steps > max_steps + 5:
            break
        action = expert.act(observation)
        if action not in env.admissible_actions():
            return {
                "ok": False,
                "seed": seed,
                "layout_id": definition.get("layout_id"),
                "structural_violations": violations,
                "expert_success": False,
                "terminal_reason": "expert_illegal_action",
                "steps": steps,
                "reward": total_reward,
                "action": action,
            }
        result = env.step(action)
        total_reward += result.reward
        observation = result.observation
        steps += 1

    success = bool(env.won and env.terminal_reason == "mission_success")
    return {
        "ok": success,
        "seed": seed,
        "layout_id": definition.get("layout_id"),
        "structural_violations": violations,
        "expert_success": success,
        "terminal_reason": env.terminal_reason,
        "steps": steps,
        "reward": total_reward,
    }


__all__ = [
    "DEFAULT_RANDOM_CONFIG",
    "generate_random_definition",
    "layout_id",
    "structural_invariants",
    "validate_definition",
]
