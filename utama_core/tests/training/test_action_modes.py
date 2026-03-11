"""Action-mode and config tests for the passing scenario."""

import math

import pytest
import torch

from utama_core.training.scenario.passing_config import (
    PassingDynamicsConfig,
    PassingResetRandomizationConfig,
    PassingRewardConfig,
    PassingScenarioConfig,
)
from utama_core.training.scenario.passing_scenario import PassingScenario
from utama_core.training.task import SSLTask


def _make_scenario(cfg: PassingScenarioConfig, batch_dim: int = 2) -> PassingScenario:
    scenario = PassingScenario()
    world = scenario.make_world(
        batch_dim=batch_dim,
        device=torch.device("cpu"),
        scenario_config=cfg,
    )
    scenario._world = world
    scenario.reset_world_at(env_index=None)
    return scenario


def _benchmark_cfg(**overrides) -> PassingScenarioConfig:
    cfg = PassingScenarioConfig(
        n_attackers=2,
        n_defenders=1,
        reset_randomization=PassingResetRandomizationConfig(benchmark_reset=True),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class TestConfigDefaults:
    def test_force_based_is_default(self):
        assert PassingDynamicsConfig().physics_mode == "force_based"

    def test_task_registry_uses_force_based_defaults(self):
        task = SSLTask.from_name("ssl_2v0_unified")
        assert task._scenario_config.dynamics.physics_mode == "force_based"

    def test_explicit_legacy_task_uses_kinematic_mode(self):
        task = SSLTask.from_name("ssl_2v0_unified_legacy")
        assert task._scenario_config.dynamics.physics_mode == "kinematic_legacy"

    def test_invalid_physics_mode_rejected(self):
        with pytest.raises(ValueError, match="physics_mode"):
            PassingScenarioConfig(dynamics=PassingDynamicsConfig(physics_mode="bad-mode"))  # type: ignore[arg-type]


class TestUnifiedMode:
    def test_unified_action_stays_four_dimensional(self):
        scenario = _make_scenario(_benchmark_cfg())
        agent = scenario.attackers[0]
        agent.action.u = torch.tensor([[1.0, 0.5, math.pi / 2, -1.0]] * 2)
        scenario.process_action(agent)
        assert agent.action.u.shape == (2, 4)
        assert not torch.isnan(agent.action.u).any()

    def test_fixed_defender_zeroes_action(self):
        scenario = _make_scenario(_benchmark_cfg())
        defender = scenario.defenders[0]
        defender.action.u = torch.ones(2, 4)
        scenario.process_action(defender)
        assert torch.allclose(defender.action.u, torch.zeros(2, 4))


class TestMacroAndLegacyModes:
    def test_macro_mode_outputs_force_triplet(self):
        cfg = PassingScenarioConfig(
            n_attackers=2,
            n_defenders=0,
            dynamics=PassingDynamicsConfig(use_macro_actions=True, use_unified_actions=False),
            rewards=PassingRewardConfig(
                passer_face_receiver_weight=0.0,
                receiver_face_ball_weight=0.0,
                kick_alignment_weight=0.0,
            ),
            reset_randomization=PassingResetRandomizationConfig(benchmark_reset=True),
        )
        scenario = _make_scenario(cfg)
        agent = scenario.attackers[0]
        agent.action.u = torch.tensor([[-0.7, 1.5, 0.0]] * 2)
        scenario.process_action(agent)
        assert agent.action.u.shape == (2, 3)
        assert not torch.isnan(agent.action.u).any()

    def test_legacy_mode_keeps_trigger_dimensions(self):
        cfg = PassingScenarioConfig(
            n_attackers=2,
            n_defenders=0,
            dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=False),
            reset_randomization=PassingResetRandomizationConfig(benchmark_reset=True),
        )
        scenario = _make_scenario(cfg)
        agent = scenario.attackers[0]
        agent.action.u = torch.tensor([[0.4, 0.0, 0.0, 1.0, 0.0, 0.0]] * 2)
        scenario.process_action(agent)
        assert agent.action.u.shape == (2, 6)
        assert not torch.isnan(agent.action.u).any()
