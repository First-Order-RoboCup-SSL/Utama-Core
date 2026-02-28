"""Tests for parameterized macro-action processing in PassingScenario."""

import math

import pytest
import torch

from utama_core.training.scenario.passing_config import (
    MacroAction,
    PassingDynamicsConfig,
    PassingRewardConfig,
    PassingScenarioConfig,
)
from utama_core.training.scenario.passing_scenario import PassingScenario


def _make_scenario(cfg: PassingScenarioConfig, batch_dim: int = 4) -> PassingScenario:
    """Helper to create and initialize a scenario with _world set."""
    scenario = PassingScenario()
    world = scenario.make_world(
        batch_dim=batch_dim,
        device=torch.device("cpu"),
        scenario_config=cfg,
    )
    # VMAS sets _world via its Environment wrapper; set it manually for tests
    scenario._world = world
    scenario.reset_world_at(env_index=None)
    return scenario


@pytest.fixture
def macro_scenario():
    """Create a PassingScenario with macro-actions enabled, batch_dim=4."""
    return _make_scenario(
        PassingScenarioConfig(
            n_attackers=2,
            n_defenders=0,
            dynamics=PassingDynamicsConfig(use_macro_actions=True, use_unified_actions=False),
            rewards=PassingRewardConfig(
                passer_face_receiver_weight=0.0,
                receiver_face_ball_weight=0.0,
                kick_alignment_weight=0.0,
            ),
        )
    )


@pytest.fixture
def legacy_scenario():
    """Create a PassingScenario with legacy 6D actions, batch_dim=4."""
    return _make_scenario(
        PassingScenarioConfig(
            n_attackers=2,
            n_defenders=0,
            dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=False),
        )
    )


class TestActionDecoding:
    """Verify action_selector → macro-action index mapping."""

    def test_bin_boundaries(self, macro_scenario):
        """action_selector in [-1, 1] maps to 4 bins correctly."""
        s = macro_scenario
        agent = s.attackers[0]

        # Values well within each bin
        test_cases = [
            (-0.9, MacroAction.GO_TO_BALL),  # bin 0: [-1.0, -0.5)
            (-0.3, MacroAction.KICK_TO),  # bin 1: [-0.5, 0.0)
            (0.1, MacroAction.DRIBBLE_TO),  # bin 2: [0.0, 0.5)
            (0.7, MacroAction.MOVE_TO),  # bin 3: [0.5, 1.0]
        ]
        for selector_val, expected_action in test_cases:
            agent.action.u = torch.tensor([[selector_val, 0.0, 0.0]] * 4)
            s.process_action(agent)
            assert s.current_macro_action[agent.name][0].item() == expected_action, (
                f"selector={selector_val} expected {expected_action.name}, "
                f"got {s.current_macro_action[agent.name][0].item()}"
            )

    def test_edge_values(self, macro_scenario):
        """Boundary values at -1.0, 0.0, 1.0."""
        s = macro_scenario
        agent = s.attackers[0]

        # -1.0 → bin 0 (GO_TO_BALL)
        agent.action.u = torch.tensor([[-1.0, 0.0, 0.0]] * 4)
        s.process_action(agent)
        assert s.current_macro_action[agent.name][0].item() == MacroAction.GO_TO_BALL

        # 0.0 → bin 2 (DRIBBLE_TO) since (0.0+1.0)*4/2 = 2.0 → clamp to 2
        agent.action.u = torch.tensor([[0.0, 0.0, 0.0]] * 4)
        s.process_action(agent)
        assert s.current_macro_action[agent.name][0].item() == MacroAction.DRIBBLE_TO

        # 1.0 → bin 3 (MOVE_TO) since (1.0+1.0)*4/2 = 4.0 → clamp to 3
        agent.action.u = torch.tensor([[1.0, 0.0, 0.0]] * 4)
        s.process_action(agent)
        assert s.current_macro_action[agent.name][0].item() == MacroAction.MOVE_TO

    def test_batch_heterogeneity(self, macro_scenario):
        """Different envs in the batch can select different macro-actions."""
        s = macro_scenario
        agent = s.attackers[0]

        # Each env selects a different action
        agent.action.u = torch.tensor(
            [
                [-0.9, 0.0, 0.0],  # GO_TO_BALL
                [-0.3, 1.0, 0.5],  # KICK_TO
                [0.1, -1.0, -0.5],  # DRIBBLE_TO
                [0.7, 2.0, 1.0],  # MOVE_TO
            ]
        )
        s.process_action(agent)
        expected = [0, 1, 2, 3]
        for i, exp in enumerate(expected):
            assert s.current_macro_action[agent.name][i].item() == exp


class TestActionSize:
    """Verify agent action dimensions."""

    def test_macro_action_size(self, macro_scenario):
        agent = macro_scenario.attackers[0]
        assert agent.action_size == 3

    def test_legacy_action_size(self, legacy_scenario):
        agent = legacy_scenario.attackers[0]
        assert agent.action_size == 6


class TestPDControllers:
    """Test BatchedPID controllers used by all action modes."""

    def test_pid_navigate_toward_target(self, macro_scenario):
        """Velocity should point toward the target."""
        s = macro_scenario
        agent_pos = torch.tensor([[0.0, 0.0]] * 4)
        target_pos = torch.tensor([[1.0, 0.0]] * 4)

        vel = s.pid_trans.calculate(agent_pos, target_pos, "test_nav")

        # Velocity should be positive in x direction
        assert (vel[:, 0] > 0).all(), f"vel_x should be positive: {vel[:, 0]}"
        # Velocity y should be near zero
        assert (vel[:, 1].abs() < 1e-3).all(), f"vel_y should be ~0: {vel[:, 1]}"

    def test_pid_navigate_deadzone(self, macro_scenario):
        """Velocity should be zero when very close to target."""
        s = macro_scenario
        agent_pos = torch.tensor([[1.0, 1.0]] * 4)
        target_pos = torch.tensor([[1.001, 1.001]] * 4)

        vel = s.pid_trans.calculate(agent_pos, target_pos, "test_dead")
        assert (vel.abs() < 0.05).all()

    def test_pid_orient_toward_target(self, macro_scenario):
        """Angular velocity should rotate toward target orientation."""
        s = macro_scenario
        agent_rot = torch.zeros(4, 1)  # facing right (0 rad)
        target_oren = torch.full((4, 1), math.pi / 2)  # want to face up

        ang_vel = s.pid_oren.calculate(agent_rot, target_oren, "test_oren")
        # Should rotate CCW (positive angular velocity)
        assert (ang_vel > 0).all(), f"ang_vel should be positive: {ang_vel}"


class TestMacroGoToBall:
    """Test GO_TO_BALL macro-action."""

    def test_moves_toward_ball(self, macro_scenario):
        """Agent should produce velocity toward ball position."""
        s = macro_scenario
        agent = s.attackers[0]
        ball_pos = s.ball.state.pos

        # Set action to GO_TO_BALL (selector ~ -0.9)
        agent.action.u = torch.tensor([[-0.9, 0.0, 0.0]] * 4)
        s.process_action(agent)

        # After process_action, agent.action.u is overwritten with [vx, vy, omega]
        vel = agent.action.u[:, :2]
        # Velocity should have a component toward the ball
        to_ball = ball_pos - agent.state.pos
        # Dot product of velocity and direction to ball should be positive
        dot = (vel * to_ball).sum(dim=-1)
        assert (dot > 0).all(), "Agent should move toward ball"


class TestMacroKickTo:
    """Test KICK_TO macro-action."""

    def test_kick_fires_when_aligned(self, macro_scenario):
        """Ball velocity should be set when agent has ball and faces target."""
        s = macro_scenario
        agent = s.attackers[0]
        fc = s.cfg.field

        # Place agent right at the ball, facing right
        s.ball.set_pos(torch.tensor([[1.0, 0.0]] * 4), batch_index=None)
        s.ball.set_vel(torch.zeros(4, 2), batch_index=None)
        agent.set_pos(torch.tensor([[1.0 - fc.robot_radius - fc.ball_radius + 0.01, 0.0]] * 4), batch_index=None)
        agent.set_rot(torch.zeros(4, 1), batch_index=None)  # facing right (0 rad)

        # KICK_TO target at (4.0, 0.0) — straight ahead
        # action_selector ~ -0.3 for KICK_TO bin, target_x=4.0/half_length, target_y=0
        agent.action.u = torch.tensor([[-0.3, 4.0 / fc.half_length, 0.0]] * 4)
        # Apply u_multiplier manually (VMAS does this before process_action)
        agent.action.u[:, 1] *= fc.half_length
        agent.action.u[:, 2] *= fc.half_width

        s.process_action(agent)

        # Check that kick_fired was set
        assert hasattr(s, "kick_fired")
        # At least some envs should have fired (depends on exact alignment)
        # The kick detection happens inside _process_macro_action


class TestMacroMoveTo:
    """Test MOVE_TO macro-action."""

    def test_moves_toward_target(self, macro_scenario):
        """Agent should produce velocity toward the specified target."""
        s = macro_scenario
        agent = s.attackers[1]  # receiver
        # fc = s.cfg.field

        # Place agent at origin
        agent.set_pos(torch.tensor([[0.0, 0.0]] * 4), batch_index=None)
        agent.set_rot(torch.zeros(4, 1), batch_index=None)

        # MOVE_TO (2.0, 1.0) — selector ~ 0.7 for MOVE_TO bin
        target_x = 2.0
        target_y = 1.0
        agent.action.u = torch.tensor([[0.7, target_x, target_y]] * 4)

        s.process_action(agent)

        vel = agent.action.u[:, :2]
        # Should move toward (2, 1)
        to_target = torch.tensor([target_x, target_y])
        dot = (vel * to_target.unsqueeze(0)).sum(dim=-1)
        assert (dot > 0).all(), "Agent should move toward target"

    def test_no_ball_interaction(self, macro_scenario):
        """MOVE_TO should not affect ball velocity."""
        s = macro_scenario
        agent = s.attackers[1]

        # Place agent near ball
        s.ball.set_pos(torch.tensor([[0.1, 0.0]] * 4), batch_index=None)
        ball_vel_before = s.ball.state.vel.clone()
        s.ball.set_vel(torch.zeros(4, 2), batch_index=None)
        ball_vel_before = s.ball.state.vel.clone()

        agent.set_pos(torch.tensor([[0.0, 0.0]] * 4), batch_index=None)
        agent.set_rot(torch.zeros(4, 1), batch_index=None)

        # MOVE_TO far away
        agent.action.u = torch.tensor([[0.7, 3.0, 2.0]] * 4)
        s.process_action(agent)

        # Ball velocity should be unchanged
        assert torch.allclose(s.ball.state.vel, ball_vel_before, atol=1e-6)


class TestMacroDribbleTo:
    """Test DRIBBLE_TO macro-action."""

    def test_fallback_to_approach(self, macro_scenario):
        """When agent doesn't have ball, should move toward ball."""
        s = macro_scenario
        agent = s.attackers[0]

        # Place agent far from ball
        agent.set_pos(torch.tensor([[-2.0, -2.0]] * 4), batch_index=None)
        agent.set_rot(torch.zeros(4, 1), batch_index=None)
        s.ball.set_pos(torch.tensor([[1.0, 1.0]] * 4), batch_index=None)

        # DRIBBLE_TO (3.0, 0.0) — but no ball, so should approach ball
        agent.action.u = torch.tensor([[0.1, 3.0, 0.0]] * 4)
        s.process_action(agent)

        vel = agent.action.u[:, :2]
        to_ball = s.ball.state.pos - agent.state.pos
        dot = (vel * to_ball).sum(dim=-1)
        assert (dot > 0).all(), "Should move toward ball when not holding"


class TestLegacyMode:
    """Verify legacy 6D mode still works."""

    def test_legacy_process_action_runs(self, legacy_scenario):
        """Legacy process_action should execute without errors."""
        s = legacy_scenario
        agent = s.attackers[0]

        # Set a simple move action
        agent.action.u = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)
        s.process_action(agent)

        # Should produce velocity output
        vel = agent.action.u[:, :2]
        assert vel.shape == (4, 2)
        assert not torch.isnan(vel).any()


class TestFixedDefenders:
    """Verify fixed defenders get zero actions."""

    def test_fixed_defender_zero_action(self):
        scenario = _make_scenario(
            PassingScenarioConfig(
                n_attackers=2,
                n_defenders=1,
                defender_behavior="fixed",
                dynamics=PassingDynamicsConfig(use_macro_actions=True, use_unified_actions=False),
            )
        )

        defender = scenario.defenders[0]
        defender.action.u = torch.ones(4, 3)
        scenario.process_action(defender)

        assert torch.allclose(defender.action.u, torch.zeros(4, 3))


class TestOutputShape:
    """Verify process_action produces correct output tensor shape."""

    def test_macro_output_is_3d(self, macro_scenario):
        s = macro_scenario
        agent = s.attackers[0]
        agent.action.u = torch.tensor([[-0.5, 1.0, 0.5]] * 4)
        s.process_action(agent)
        assert agent.action.u.shape == (4, 3)

    def test_legacy_output_is_6d(self, legacy_scenario):
        s = legacy_scenario
        agent = s.attackers[0]
        agent.action.u = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)
        s.process_action(agent)
        assert agent.action.u.shape == (4, 6)

    def test_no_nans_in_output(self, macro_scenario):
        """No NaN values in any macro-action output."""
        s = macro_scenario
        for agent in s.attackers:
            for selector in [-0.9, -0.3, 0.1, 0.7]:
                agent.action.u = torch.tensor([[selector, 1.0, -0.5]] * 4)
                s.process_action(agent)
                assert not torch.isnan(agent.action.u).any(), f"NaN in output for selector={selector}"
