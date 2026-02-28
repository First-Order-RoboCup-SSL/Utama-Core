"""Tests for unified action space processing in PassingScenario."""

import math

import pytest
import torch

from utama_core.training.scenario.batched_pid import (
    BatchedOrientationPID,
    BatchedTranslationPID,
)
from utama_core.training.scenario.passing_config import (
    PassingDynamicsConfig,
    PassingRewardConfig,
    PassingScenarioConfig,
)
from utama_core.training.scenario.passing_scenario import PassingScenario

BATCH = 4


def _make_scenario(cfg: PassingScenarioConfig, batch_dim: int = BATCH) -> PassingScenario:
    """Helper to create and initialize a scenario with _world set."""
    scenario = PassingScenario()
    world = scenario.make_world(
        batch_dim=batch_dim,
        device=torch.device("cpu"),
        scenario_config=cfg,
    )
    scenario._world = world
    scenario.reset_world_at(env_index=None)
    return scenario


def _unified_cfg(**overrides) -> PassingScenarioConfig:
    dyn = PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=True)
    rew = PassingRewardConfig(
        passer_face_receiver_weight=0.0,
        receiver_face_ball_weight=0.0,
        kick_alignment_weight=0.0,
    )
    defaults = dict(
        n_attackers=2,
        n_defenders=0,
        dynamics=dyn,
        rewards=rew,
    )
    defaults.update(overrides)
    return PassingScenarioConfig(**defaults)


@pytest.fixture
def unified_scenario():
    return _make_scenario(_unified_cfg())


@pytest.fixture
def unified_1v1():
    return _make_scenario(_unified_cfg(n_defenders=1))


# ------------------------------------------------------------------
# Action space setup
# ------------------------------------------------------------------


class TestActionSpaceSetup:
    def test_action_size_is_4(self, unified_scenario):
        for agent in unified_scenario.attackers:
            assert agent.action_size == 4

    def test_u_multiplier(self, unified_scenario):
        s = unified_scenario
        fc = s.cfg.field
        agent = s.attackers[0]
        expected = [fc.half_length, fc.half_width, math.pi, 1.0]
        for i, val in enumerate(expected):
            assert agent.u_multiplier[i] == pytest.approx(val)

    def test_output_shape(self, unified_scenario):
        s = unified_scenario
        agent = s.attackers[0]
        agent.action.u = torch.zeros(BATCH, 4)
        s.process_action(agent)
        # Output is [vx, vy, ang_vel] = 3D (PID output for VelocityHolonomic)
        assert agent.action.u.shape == (BATCH, 3)

    def test_no_nans(self, unified_scenario):
        s = unified_scenario
        for agent in s.attackers:
            for _ in range(5):
                agent.action.u = torch.randn(BATCH, 4)
                s.process_action(agent)
                assert not torch.isnan(agent.action.u).any()


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------


class TestNavigation:
    def test_moves_toward_target(self, unified_scenario):
        s = unified_scenario
        agent = s.attackers[1]
        agent.set_pos(torch.tensor([[0.0, 0.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)

        # Target at (2.0, 1.0), target_oren=0, kick_intent < 0
        agent.action.u = torch.tensor([[2.0, 1.0, 0.0, -1.0]] * BATCH)
        s.process_action(agent)

        vel = agent.action.u[:, :2]
        to_target = torch.tensor([2.0, 1.0])
        dot = (vel * to_target.unsqueeze(0)).sum(dim=-1)
        assert (dot > 0).all(), f"Agent should move toward target, vel={vel[0]}"

    def test_deadzone_at_target(self, unified_scenario):
        s = unified_scenario
        agent = s.attackers[1]
        agent.set_pos(torch.tensor([[1.0, 1.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)

        # Target essentially at agent position
        agent.action.u = torch.tensor([[1.001, 1.001, 0.0, -1.0]] * BATCH)
        s.process_action(agent)

        vel = agent.action.u[:, :2]
        assert (vel.abs() < 0.05).all(), f"Velocity should be ~0 at target: {vel[0]}"


# ------------------------------------------------------------------
# Orientation
# ------------------------------------------------------------------


class TestOrientation:
    def test_orients_toward_target_oren(self, unified_scenario):
        s = unified_scenario
        agent = s.attackers[1]
        agent.set_pos(torch.tensor([[0.0, 0.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)  # facing right (0 rad)

        # target_oren = pi/2 — should rotate CCW (positive angular vel)
        agent.action.u = torch.tensor([[1.0, 0.0, math.pi / 2, -1.0]] * BATCH)
        s.process_action(agent)

        ang = agent.action.u[:, 2]
        assert (ang > 0).all(), f"Should rotate CCW toward target_oren: {ang[0]}"

    def test_orientation_deadzone(self, unified_scenario):
        s = unified_scenario
        agent = s.attackers[1]
        agent.set_pos(torch.tensor([[0.0, 0.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)  # facing right (0 rad)

        # target_oren = 0.0 — already facing right, angular vel should be ~0
        agent.action.u = torch.tensor([[1.0, 0.0, 0.0, -1.0]] * BATCH)
        s.process_action(agent)

        ang = agent.action.u[:, 2]
        assert (ang.abs() < 0.5).all(), f"Angular vel should be small: {ang[0]}"


# ------------------------------------------------------------------
# Auto-dribble
# ------------------------------------------------------------------


class TestAutoDribble:
    def test_dribble_engages_when_has_ball(self, unified_scenario):
        """Ball velocity should change (attract force) when agent has ball."""
        s = unified_scenario
        agent = s.attackers[0]
        fc = s.cfg.field

        # Place agent at (0,0) facing right, ball just in front
        agent.set_pos(torch.tensor([[0.0, 0.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)
        ball_x = fc.robot_radius + fc.ball_radius - 0.01
        s.ball.set_pos(torch.tensor([[ball_x, 0.0]] * BATCH), batch_index=None)
        s.ball.set_vel(torch.zeros(BATCH, 2), batch_index=None)

        # No kick intent, target_oren=0 (facing right)
        agent.action.u = torch.tensor([[2.0, 0.0, 0.0, -1.0]] * BATCH)
        s.process_action(agent)

        ball_vel = s.ball.state.vel
        # Ball should have been attracted (vel changed from zero)
        assert ball_vel.abs().sum() > 0, "Dribble should attract ball"

    def test_no_dribble_when_far(self, unified_scenario):
        s = unified_scenario
        agent = s.attackers[0]

        agent.set_pos(torch.tensor([[-2.0, -2.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)
        s.ball.set_pos(torch.tensor([[1.0, 1.0]] * BATCH), batch_index=None)
        s.ball.set_vel(torch.zeros(BATCH, 2), batch_index=None)

        agent.action.u = torch.tensor([[1.0, 1.0, 0.0, -1.0]] * BATCH)
        s.process_action(agent)

        assert torch.allclose(s.ball.state.vel, torch.zeros(BATCH, 2), atol=1e-6)


# ------------------------------------------------------------------
# Kick (2-frame dribbler release)
# ------------------------------------------------------------------


class TestKick:
    def _setup_aligned_kick(self, scenario):
        """Place agent at ball, facing right, ball in front. Returns agent."""
        s = scenario
        agent = s.attackers[0]
        fc = s.cfg.field
        offset = fc.robot_radius + fc.ball_radius - 0.01

        agent.set_pos(torch.tensor([[0.0, 0.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)
        s.ball.set_pos(torch.tensor([[offset, 0.0]] * BATCH), batch_index=None)
        s.ball.set_vel(torch.zeros(BATCH, 2), batch_index=None)
        return agent

    def test_kick_not_immediate(self, unified_scenario):
        """First frame with kick_intent should NOT fire kick (dribbler release frame)."""
        s = unified_scenario
        agent = self._setup_aligned_kick(s)

        # Target at (4, 0) — aligned with facing (right)
        agent.action.u = torch.tensor([[4.0, 0.0, 0.0, 1.0]] * BATCH)
        s.process_action(agent)

        # Ball velocity should NOT be kick_speed yet (pending frame)
        ball_speed = torch.norm(s.ball.state.vel, dim=-1)
        assert (
            ball_speed < s.cfg.dynamics.kick_speed - 0.1
        ).all(), f"Kick should not fire on first frame, ball_speed={ball_speed[0]}"

    def test_kick_fires_on_second_frame(self, unified_scenario):
        """Second frame should fire the kick."""
        s = unified_scenario
        agent = self._setup_aligned_kick(s)

        # Frame 1: set pending
        agent.action.u = torch.tensor([[4.0, 0.0, 0.0, 1.0]] * BATCH)
        s.process_action(agent)

        # Re-place agent/ball for frame 2 (ball may have drifted slightly)
        self._setup_aligned_kick(s)

        # Frame 2: fire kick
        agent.action.u = torch.tensor([[4.0, 0.0, 0.0, 1.0]] * BATCH)
        s.process_action(agent)

        ball_speed = torch.norm(s.ball.state.vel, dim=-1)
        assert (
            ball_speed > s.cfg.dynamics.kick_speed - 0.5
        ).all(), f"Kick should fire on second frame, ball_speed={ball_speed[0]}"

    def test_kick_blocked_without_ball(self, unified_scenario):
        """No kick when agent doesn't have ball."""
        s = unified_scenario
        agent = s.attackers[0]

        agent.set_pos(torch.tensor([[-2.0, 0.0]] * BATCH), batch_index=None)
        agent.set_rot(torch.zeros(BATCH, 1), batch_index=None)
        s.ball.set_pos(torch.tensor([[1.0, 0.0]] * BATCH), batch_index=None)
        s.ball.set_vel(torch.zeros(BATCH, 2), batch_index=None)

        agent.action.u = torch.tensor([[4.0, 0.0, 0.0, 1.0]] * BATCH)
        s.process_action(agent)

        assert torch.allclose(s.ball.state.vel, torch.zeros(BATCH, 2), atol=1e-6)

    def test_kick_blocked_when_misaligned(self, unified_scenario):
        """No kick pending when target is behind robot (misaligned)."""
        s = unified_scenario
        agent = self._setup_aligned_kick(s)

        # Target behind robot (facing right but target at negative x)
        agent.action.u = torch.tensor([[-4.0, 0.0, 0.0, 1.0]] * BATCH)
        s.process_action(agent)

        # kick_pending should NOT be set
        assert not s.kick_pending[agent.name].any()

    def test_kick_pending_resets(self, unified_scenario):
        """kick_pending clears when kick_intent goes negative."""
        s = unified_scenario
        agent = self._setup_aligned_kick(s)

        # Frame 1: set pending
        agent.action.u = torch.tensor([[4.0, 0.0, 0.0, 1.0]] * BATCH)
        s.process_action(agent)
        assert s.kick_pending[agent.name].any() or s.kick_fired[agent.name].any()

        # Frame 2: kick_intent negative → clear pending
        self._setup_aligned_kick(s)
        agent.action.u = torch.tensor([[4.0, 0.0, 0.0, -1.0]] * BATCH)
        s.process_action(agent)

        assert not s.kick_pending[agent.name].any()

    def test_kick_fired_populated(self, unified_scenario):
        """kick_fired dict should be populated for reward computation."""
        s = unified_scenario
        agent = s.attackers[0]
        agent.action.u = torch.zeros(BATCH, 4)
        s.process_action(agent)

        assert hasattr(s, "kick_fired")
        assert agent.name in s.kick_fired
        assert s.kick_fired[agent.name].shape == (BATCH,)


# ------------------------------------------------------------------
# Fixed defenders
# ------------------------------------------------------------------


class TestFixedDefenders:
    def test_fixed_defender_zero_action(self, unified_1v1):
        s = unified_1v1
        defender = s.defenders[0]
        defender.action.u = torch.ones(BATCH, 4)
        s.process_action(defender)
        assert torch.allclose(defender.action.u, torch.zeros(BATCH, 4))


# ------------------------------------------------------------------
# BatchedPID
# ------------------------------------------------------------------


class TestBatchedPID:
    def test_translation_toward_target(self):
        pid = BatchedTranslationPID()
        current = torch.tensor([[0.0, 0.0]] * BATCH)
        target = torch.tensor([[1.0, 0.0]] * BATCH)

        vel = pid.calculate(current, target, "test")
        assert (vel[:, 0] > 0).all()
        assert (vel[:, 1].abs() < 1e-3).all()

    def test_translation_deadzone(self):
        pid = BatchedTranslationPID()
        current = torch.tensor([[1.0, 1.0]] * BATCH)
        target = torch.tensor([[1.001, 1.001]] * BATCH)

        vel = pid.calculate(current, target, "test")
        assert (vel.abs() < 0.05).all()

    def test_orientation_toward_target(self):
        pid = BatchedOrientationPID()
        current = torch.zeros(BATCH, 1)
        target = torch.full((BATCH, 1), math.pi / 2)

        ang = pid.calculate(current, target, "test")
        assert (ang > 0).all()

    def test_orientation_deadzone(self):
        pid = BatchedOrientationPID()
        current = torch.zeros(BATCH, 1)
        target = torch.full((BATCH, 1), 0.0005)

        ang = pid.calculate(current, target, "test")
        assert (ang.abs() < 0.01).all()

    def test_reset(self):
        pid = BatchedTranslationPID()
        current = torch.tensor([[0.0, 0.0]] * BATCH)
        target = torch.tensor([[1.0, 0.0]] * BATCH)

        pid.calculate(current, target, "test")
        pid.reset("test")
        vel = pid.calculate(current, target, "test")
        # Should still work after reset
        assert (vel[:, 0] > 0).all()


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------


class TestConfigValidation:
    def test_rejects_both_modes(self):
        with pytest.raises(ValueError, match="Cannot enable both"):
            PassingScenarioConfig(
                dynamics=PassingDynamicsConfig(
                    use_macro_actions=True,
                    use_unified_actions=True,
                ),
            )

    def test_legacy_mode_accepted(self):
        cfg = PassingScenarioConfig(
            dynamics=PassingDynamicsConfig(
                use_macro_actions=False,
                use_unified_actions=False,
            ),
        )
        assert cfg.dynamics.use_macro_actions is False
        assert cfg.dynamics.use_unified_actions is False


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


class TestBackwardCompat:
    def test_macro_still_works(self):
        cfg = PassingScenarioConfig(
            n_attackers=2,
            n_defenders=0,
            dynamics=PassingDynamicsConfig(use_macro_actions=True, use_unified_actions=False),
            rewards=PassingRewardConfig(
                passer_face_receiver_weight=0.0,
                receiver_face_ball_weight=0.0,
                kick_alignment_weight=0.0,
            ),
        )
        s = _make_scenario(cfg)
        agent = s.attackers[0]
        assert agent.action_size == 3
        agent.action.u = torch.tensor([[-0.5, 1.0, 0.5]] * BATCH)
        s.process_action(agent)
        assert agent.action.u.shape == (BATCH, 3)
        assert not torch.isnan(agent.action.u).any()

    def test_legacy_still_works(self):
        cfg = PassingScenarioConfig(
            n_attackers=2,
            n_defenders=0,
            dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=False),
        )
        s = _make_scenario(cfg)
        agent = s.attackers[0]
        assert agent.action_size == 6
        agent.action.u = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]] * BATCH)
        s.process_action(agent)
        assert agent.action.u.shape == (BATCH, 6)
        assert not torch.isnan(agent.action.u).any()
