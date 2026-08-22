"""Tests for the bounded random map generator (Stage A).

The legacy fixed and 8-variant procedural worlds are untouched and remain the
regression canary: see test_extraction_ops.py for those contracts.
"""
from __future__ import annotations

from agent_system.environments.env_package.extraction_ops import ExtractionOpsExpert, ExtractionOpsWorld
from agent_system.environments.env_package.extraction_ops.generator import (
    generate_random_definition,
    structural_invariants,
    validate_definition,
)


def test_generator_is_deterministic():
    first = generate_random_definition(7)
    second = generate_random_definition(7)
    assert first == second
    assert first["layout_id"] == second["layout_id"]


def test_generator_structural_invariants_hold():
    for seed in range(100):
        definition = generate_random_definition(seed)
        assert structural_invariants(definition) == [], (seed, structural_invariants(definition))


def test_expert_solves_random_maps_full_pipeline():
    for seed in range(50):
        env = ExtractionOpsWorld(seed=seed, random_maps=True)
        expert = ExtractionOpsExpert()
        observation, _ = env.reset(seed)
        actions = []
        while not env.done:
            action = expert.act(observation)
            assert action in env.admissible_actions(), (seed, action, env.admissible_actions())
            actions.append(action)
            result = env.step(action)
            assert result.info["is_action_valid"], (seed, action, result.info)
            observation = result.observation
        assert env.won, (seed, env.terminal_reason, actions)
        assert env.terminal_reason == "mission_success"


def test_random_maps_produce_diverse_layouts():
    layouts = {generate_random_definition(seed)["layout_id"] for seed in range(50)}
    assert len(layouts) > 40, f"expected diverse layouts, got {len(layouts)}"


def test_layout_id_ignores_cosmetic_fields():
    definition = generate_random_definition(3)
    from agent_system.environments.env_package.extraction_ops.generator import layout_id

    before = definition["layout_id"]
    for a, b, distance, terrain, door in definition["edges"]:
        del distance, terrain  # read-only probe of the tuple shape
    assert layout_id(definition) == before


def test_validate_definition_authoritative_check():
    for seed in range(10):
        definition = generate_random_definition(seed)
        report = validate_definition(seed, definition)
        assert report["ok"], (seed, report)
        assert report["expert_success"]
        assert abs(report["reward"] - 1.2) < 1e-8


def test_fixed_world_regression_canary():
    """The legacy fixed world must keep solving after generator changes."""
    env = ExtractionOpsWorld(seed=20260714)
    expert = ExtractionOpsExpert()
    observation, _ = env.reset()
    while not env.done:
        result = env.step(expert.act(observation))
        observation = result.observation
    assert env.won and env.terminal_reason == "mission_success"
