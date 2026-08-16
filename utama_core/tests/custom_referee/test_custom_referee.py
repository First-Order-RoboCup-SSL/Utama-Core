"""Unit tests for the CustomReferee system."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.custom_referee import CustomReferee
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.custom_referee.rules.ball_speed_rule import BallSpeedRule
from utama_core.custom_referee.rules.defense_area_rule import DefenseAreaRule
from utama_core.custom_referee.rules.goal_rule import GoalRule
from utama_core.custom_referee.rules.keep_out_rule import KeepOutRule
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)


def _ball(x: float, y: float, z: float = 0.0, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, z), v=Vector3D(vx, vy, vz), a=Vector3D(0, 0, 0))


def _robot(robot_id: int, x: float, y: float, is_friendly: bool, has_ball: bool = False) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _frame(
    ball: Ball,
    friendly_robots: dict | None = None,
    enemy_robots: dict | None = None,
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = False,
    ts: float = 10.0,
) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly_robots or {},
        enemy_robots=enemy_robots or {},
        ball=ball,
        referee=None,
    )


def _state_machine() -> GameStateMachine:
    sm = GameStateMachine(
        half_duration_seconds=300.0,
        kickoff_team="yellow",
        n_robots_yellow=3,
        n_robots_blue=3,
    )
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.NORMAL_START, 0.0)
    return sm


# ---------------------------------------------------------------------------
# GoalRule
# ---------------------------------------------------------------------------


class TestGoalRule:
    def test_right_goal_blue_scores_when_yellow_is_right(self):
        # Yellow defends right goal → ball in right goal → blue scored → yellow kicks off.
        rule = GoalRule(cooldown_seconds=1.0)
        frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.rule_name == "goal"
        assert violation.status_message == "Goal by Blue"
        assert violation.next_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    def test_left_goal_yellow_scores_when_yellow_is_right(self):
        # Blue defends left goal → ball in left goal → yellow scored → blue kicks off.
        rule = GoalRule(cooldown_seconds=1.0)
        frame = _frame(ball=_ball(-5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.status_message == "Goal by Yellow"
        assert violation.next_command == RefereeCommand.PREPARE_KICKOFF_BLUE

    def test_right_goal_yellow_scores_when_yellow_is_left(self):
        # Blue defends right goal → ball in right goal → yellow scored → blue kicks off.
        rule = GoalRule(cooldown_seconds=1.0)
        frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=False)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.status_message == "Goal by Yellow"
        assert violation.next_command == RefereeCommand.PREPARE_KICKOFF_BLUE

    def test_no_goal_wide_shot(self):
        rule = GoalRule()
        frame = _frame(ball=_ball(5.0, 1.0))  # y=1.0 > half_goal_width=0.5
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None

    def test_goal_respects_cooldown(self):
        rule = GoalRule(cooldown_seconds=2.0)
        frame1 = _frame(ball=_ball(5.0, 0.0), my_team_is_right=True, ts=10.0)
        v1 = rule.check(frame1, GEO, RefereeCommand.NORMAL_START)
        assert v1 is not None

        # Second detection within cooldown window — must be suppressed.
        frame2 = _frame(ball=_ball(5.0, 0.0), my_team_is_right=True, ts=10.5)
        v2 = rule.check(frame2, GEO, RefereeCommand.NORMAL_START)
        assert v2 is None

        # After cooldown expires — should fire again.
        frame3 = _frame(ball=_ball(5.0, 0.0), my_team_is_right=True, ts=13.0)
        v3 = rule.check(frame3, GEO, RefereeCommand.NORMAL_START)
        assert v3 is not None

    def test_no_detection_during_stop(self):
        rule = GoalRule()
        frame = _frame(ball=_ball(5.0, 0.0))
        assert rule.check(frame, GEO, RefereeCommand.STOP) is None


# ---------------------------------------------------------------------------
# OutOfBoundsRule
# ---------------------------------------------------------------------------


class TestOutOfBoundsRule:
    def test_ball_out_top(self):
        rule = OutOfBoundsRule()
        frame = _frame(ball=_ball(0.0, 3.5), my_team_is_yellow=True)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.rule_name == "out_of_bounds"

    def test_ball_out_right_side(self):
        rule = OutOfBoundsRule()
        frame = _frame(ball=_ball(5.0, 1.0))  # wide — not in goal (y=1.0 > 0.5)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None

    def test_ball_in_field_no_violation(self):
        rule = OutOfBoundsRule()
        frame = _frame(ball=_ball(0.0, 0.0))
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None

    def test_ball_in_goal_no_out_of_bounds(self):
        rule = OutOfBoundsRule()
        frame = _frame(ball=_ball(5.0, 0.0))  # in right goal
        # GoalRule handles this; OutOfBoundsRule must not also fire.
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None

    def test_free_kick_assigned_to_non_touching_team_friendly_yellow(self):
        """Friendly (yellow) last touched → enemy (blue) gets free kick."""
        rule = OutOfBoundsRule()
        friendly = {0: _robot(0, 4.4, 2.9, is_friendly=True, has_ball=True)}
        frame_before = _frame(ball=_ball(4.4, 2.9), friendly_robots=friendly, my_team_is_yellow=True, ts=9.9)
        rule.check(frame_before, GEO, RefereeCommand.NORMAL_START)

        frame_out = _frame(ball=_ball(0.0, 3.5), my_team_is_yellow=True, ts=10.0)
        violation = rule.check(frame_out, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.next_command == RefereeCommand.DIRECT_FREE_BLUE

    def test_designated_position_is_infield(self):
        rule = OutOfBoundsRule()
        frame = _frame(ball=_ball(0.0, 3.5))
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        px, py = violation.designated_position
        assert abs(py) < GEO.half_width  # placed infield


# ---------------------------------------------------------------------------
# BallSpeedRule
# ---------------------------------------------------------------------------


class TestBallSpeedRule:
    def test_fires_when_ball_exceeds_speed_limit(self):
        rule = BallSpeedRule(max_speed_mps=6.5)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True, has_ball=True)}
        frame = _frame(ball=_ball(0.0, 0.0, vx=7.0, vy=0.0), friendly_robots=friendly, my_team_is_yellow=True)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.rule_name == "ball_speed"

    def test_no_violation_below_speed_limit(self):
        rule = BallSpeedRule(max_speed_mps=6.5)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True, has_ball=True)}
        frame = _frame(ball=_ball(0.0, 0.0, vx=3.0, vy=0.0), friendly_robots=friendly, my_team_is_yellow=True)
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None

    def test_z_velocity_from_a_bounce_does_not_count(self):
        """A vertical bounce should not trigger the ground-speed limit."""
        rule = BallSpeedRule(max_speed_mps=6.5)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True, has_ball=True)}
        frame = _frame(ball=_ball(0.0, 0.0, vx=1.0, vy=0.0, vz=8.0), friendly_robots=friendly, my_team_is_yellow=True)
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None

    def test_only_fires_once_per_kick_not_every_frame(self):
        rule = BallSpeedRule(max_speed_mps=6.5)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True, has_ball=True)}
        frame = _frame(ball=_ball(0.0, 0.0, vx=7.0, vy=0.0), friendly_robots=friendly, my_team_is_yellow=True)
        first = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        second = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert first is not None
        assert second is None  # still fast, but not a new kick

    def test_fires_again_after_dropping_below_and_back_above_limit(self):
        rule = BallSpeedRule(max_speed_mps=6.5)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True, has_ball=True)}
        fast = _frame(ball=_ball(0.0, 0.0, vx=7.0, vy=0.0), friendly_robots=friendly, my_team_is_yellow=True)
        slow = _frame(ball=_ball(0.0, 0.0, vx=1.0, vy=0.0), friendly_robots=friendly, my_team_is_yellow=True)

        assert rule.check(fast, GEO, RefereeCommand.NORMAL_START) is not None
        assert rule.check(slow, GEO, RefereeCommand.NORMAL_START) is None
        assert rule.check(fast, GEO, RefereeCommand.NORMAL_START) is not None

    def test_free_kick_assigned_to_non_kicking_team(self):
        """Friendly (yellow) kicked → enemy (blue) gets the free kick."""
        rule = BallSpeedRule(max_speed_mps=6.5)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True, has_ball=True)}
        frame = _frame(ball=_ball(0.0, 0.0, vx=7.0, vy=0.0), friendly_robots=friendly, my_team_is_yellow=True)
        violation = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert violation is not None
        assert violation.next_command == RefereeCommand.DIRECT_FREE_BLUE

    def test_inactive_outside_active_play(self):
        rule = BallSpeedRule(max_speed_mps=6.5)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True, has_ball=True)}
        frame = _frame(ball=_ball(0.0, 0.0, vx=7.0, vy=0.0), friendly_robots=friendly, my_team_is_yellow=True)
        assert rule.check(frame, GEO, RefereeCommand.STOP) is None

    def test_no_violation_without_a_known_last_touch(self):
        """No robot has ever registered a touch — nothing to attribute the kick to."""
        rule = BallSpeedRule(max_speed_mps=6.5)
        frame = _frame(ball=_ball(0.0, 0.0, vx=7.0, vy=0.0))
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None


# ---------------------------------------------------------------------------
# DefenseAreaRule
# ---------------------------------------------------------------------------


class TestDefenseAreaRule:
    def _frame_with_attacker_in_defense(self, my_team_is_right: bool = False) -> GameFrame:
        # Enemy robot inside my (left) defense area.
        enemy = {0: _robot(0, -4.3, 0.5, is_friendly=False)}
        return _frame(
            ball=_ball(0, 0),
            enemy_robots=enemy,
            my_team_is_right=my_team_is_right,
        )

    def test_fires_during_normal_start(self):
        rule = DefenseAreaRule()
        frame = self._frame_with_attacker_in_defense(my_team_is_right=False)
        v = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.rule_name == "defense_area"

    def test_does_not_fire_during_stop(self):
        rule = DefenseAreaRule()
        frame = self._frame_with_attacker_in_defense()
        assert rule.check(frame, GEO, RefereeCommand.STOP) is None

    def test_does_not_fire_during_force_start_no_actually_fires(self):
        rule = DefenseAreaRule()
        frame = self._frame_with_attacker_in_defense(my_team_is_right=False)
        v = rule.check(frame, GEO, RefereeCommand.FORCE_START)
        assert v is not None

    def test_too_many_defenders(self):
        rule = DefenseAreaRule(max_defenders=1)
        # Two friendly robots in own (left) defense area.
        friendly = {
            0: _robot(0, -4.3, 0.0, is_friendly=True),
            1: _robot(1, -4.3, 0.5, is_friendly=True),
        }
        frame = _frame(ball=_ball(0, 0), friendly_robots=friendly, my_team_is_right=False, my_team_is_yellow=True)
        v = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.next_command == RefereeCommand.DIRECT_FREE_BLUE


# ---------------------------------------------------------------------------
# KeepOutRule
# ---------------------------------------------------------------------------


class TestKeepOutRule:
    def test_no_trigger_before_persistence_threshold(self):
        rule = KeepOutRule(radius_meters=0.5, violation_persistence_frames=5)
        friendly = {0: _robot(0, 0.2, 0.0, is_friendly=True)}
        # Enemy is kicking → check friendly.
        for _ in range(4):
            frame = _frame(ball=_ball(0.0, 0.0), friendly_robots=friendly)
            v = rule.check(frame, GEO, RefereeCommand.DIRECT_FREE_BLUE)
            assert v is None

    def test_triggers_after_persistence_threshold(self):
        rule = KeepOutRule(radius_meters=0.5, violation_persistence_frames=5)
        friendly = {0: _robot(0, 0.2, 0.0, is_friendly=True)}
        v = None
        for _ in range(5):
            frame = _frame(ball=_ball(0.0, 0.0), friendly_robots=friendly)
            v = rule.check(frame, GEO, RefereeCommand.DIRECT_FREE_BLUE)
        assert v is not None
        assert v.rule_name == "keep_out"

    def test_resets_on_non_violation_frame(self):
        rule = KeepOutRule(radius_meters=0.5, violation_persistence_frames=5)
        friendly = {0: _robot(0, 0.2, 0.0, is_friendly=True)}
        for _ in range(4):
            frame = _frame(ball=_ball(0.0, 0.0), friendly_robots=friendly)
            rule.check(frame, GEO, RefereeCommand.DIRECT_FREE_BLUE)

        # Robot moves away — count resets.
        far_frame = _frame(
            ball=_ball(0.0, 0.0),
            friendly_robots={0: _robot(0, 2.0, 0.0, is_friendly=True)},
        )
        rule.check(far_frame, GEO, RefereeCommand.DIRECT_FREE_BLUE)

        # Needs another full persistence run to trigger.
        v = None
        for _ in range(5):
            frame = _frame(ball=_ball(0.0, 0.0), friendly_robots=friendly)
            v = rule.check(frame, GEO, RefereeCommand.DIRECT_FREE_BLUE)
        assert v is not None

    def test_inactive_during_normal_start(self):
        rule = KeepOutRule(radius_meters=0.5, violation_persistence_frames=1)
        friendly = {0: _robot(0, 0.1, 0.0, is_friendly=True)}
        frame = _frame(ball=_ball(0.0, 0.0), friendly_robots=friendly)
        assert rule.check(frame, GEO, RefereeCommand.NORMAL_START) is None


# ---------------------------------------------------------------------------
# GameStateMachine
# ---------------------------------------------------------------------------


class TestGameStateMachine:
    def test_goal_increments_yellow_score(self):
        from utama_core.custom_referee.rules.base_rule import RuleViolation

        sm = _state_machine()
        violation = RuleViolation(
            rule_name="goal",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
            status_message="Goal by Yellow",
        )
        data = sm.step(current_time=10.0, violation=violation)
        assert sm.yellow_team.score == 1
        assert sm.blue_team.score == 0
        assert data.referee_command == RefereeCommand.STOP
        assert data.next_command == RefereeCommand.BALL_PLACEMENT_BLUE

    def test_goal_increments_blue_score(self):
        from utama_core.custom_referee.rules.base_rule import RuleViolation

        sm = _state_machine()
        violation = RuleViolation(
            rule_name="goal",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
            status_message="Goal by Blue",
        )
        sm.step(current_time=10.0, violation=violation)
        assert sm.blue_team.score == 1
        assert sm.yellow_team.score == 0

    def test_transition_cooldown_suppresses_duplicate(self):
        from utama_core.custom_referee.rules.base_rule import RuleViolation

        sm = _state_machine()
        violation = RuleViolation(
            rule_name="goal",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
            status_message="Goal",
        )
        sm.step(current_time=10.0, violation=violation)
        assert sm.yellow_team.score == 1

        # Second goal within cooldown window — must be suppressed.
        sm.step(current_time=10.1, violation=violation)
        assert sm.yellow_team.score == 1  # still 1

    def test_goal_sets_designated_position_to_centre(self):
        from utama_core.custom_referee.rules.base_rule import RuleViolation

        sm = _state_machine()
        violation = RuleViolation(
            rule_name="goal",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
            status_message="Goal by Yellow",
        )
        data = sm.step(current_time=10.0, violation=violation)
        assert data.designated_position == (0.0, 0.0)

    def test_goal_status_message_is_propagated_into_referee_data(self):
        from utama_core.custom_referee.rules.base_rule import RuleViolation

        sm = _state_machine()
        violation = RuleViolation(
            rule_name="goal",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
            status_message="Goal by Yellow",
        )
        data = sm.step(current_time=10.0, violation=violation)
        assert data.status_message == "Goal by Yellow"

    def test_manual_command_clears_status_message(self):
        from utama_core.custom_referee.rules.base_rule import RuleViolation

        sm = _state_machine()
        violation = RuleViolation(
            rule_name="goal",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
            status_message="Goal by Yellow",
        )
        sm.step(current_time=10.0, violation=violation)

        sm.set_command(RefereeCommand.NORMAL_START, timestamp=11.0)
        data = sm.step(current_time=11.0, violation=None)
        assert data.status_message is None

    def test_manual_set_command(self):
        sm = _state_machine()
        sm.set_command(RefereeCommand.NORMAL_START, timestamp=5.0)
        assert sm.command == RefereeCommand.NORMAL_START

    def test_manual_ball_placement_auto_advances_from_stop_and_then_to_normal_start(self):
        sm = _state_machine()
        sm.set_command(RefereeCommand.BALL_PLACEMENT_YELLOW, timestamp=1.0)
        sm.ball_placement_target = (0.0, 0.0)

        clear_frame = _frame(ball=_ball(2.0, 0.0), ts=10.0)
        data = sm.step(current_time=10.0, violation=None, game_frame=clear_frame)
        assert data.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW
        assert data.next_command == RefereeCommand.NORMAL_START

        placed_frame = _frame(ball=_ball(0.0, 0.0), ts=20.0)
        sm.step(current_time=20.0, violation=None, game_frame=placed_frame)
        data = sm.step(current_time=23.0, violation=None, game_frame=placed_frame)
        assert data.referee_command == RefereeCommand.NORMAL_START
        assert data.next_command is None

    def test_manual_penalty_auto_advances_from_stop_and_then_to_normal_start(self):
        sm = _state_machine()
        sm.set_command(RefereeCommand.PREPARE_PENALTY_YELLOW, timestamp=1.0)

        clear_frame = _frame(ball=_ball(0.0, 0.0), ts=10.0)
        data = sm.step(current_time=10.0, violation=None, game_frame=clear_frame)
        assert data.referee_command == RefereeCommand.PREPARE_PENALTY_YELLOW
        assert data.next_command == RefereeCommand.NORMAL_START

        ready_attackers = {0: _robot(0, 2.25, 0.0, is_friendly=True)}
        ready_frame = _frame(ball=_ball(0.0, 0.0), friendly_robots=ready_attackers, ts=14.0)
        sm.step(current_time=14.0, violation=None, game_frame=ready_frame)
        data = sm.step(current_time=17.0, violation=None, game_frame=ready_frame)
        assert data.referee_command == RefereeCommand.NORMAL_START
        assert data.next_command is None

    def test_reset_restores_score_command_and_stage(self):
        from utama_core.custom_referee.rules.base_rule import RuleViolation

        sm = _state_machine()
        violation = RuleViolation(
            rule_name="goal",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
            status_message="Goal by Yellow",
        )
        sm.step(current_time=10.0, violation=violation)
        assert sm.yellow_team.score == 1
        assert sm.command == RefereeCommand.STOP

        sm.reset()
        assert sm.yellow_team.score == 0
        assert sm.blue_team.score == 0
        assert sm.command == RefereeCommand.HALT
        assert sm.stage == Stage.NORMAL_FIRST_HALF_PRE
        assert sm.next_command is None
        assert sm.ball_placement_target is None

    def test_reset_clears_auto_advance_timers(self):
        sm = _state_machine()
        sm.set_command(RefereeCommand.PREPARE_PENALTY_YELLOW, timestamp=1.0)
        clear_frame = _frame(ball=_ball(0.0, 0.0), ts=1.5)
        sm.step(current_time=1.5, violation=None, game_frame=clear_frame)  # STOP → PREPARE_PENALTY_YELLOW

        ready_attackers = {0: _robot(0, 2.25, 0.0, is_friendly=True)}
        ready_frame = _frame(ball=_ball(0.0, 0.0), friendly_robots=ready_attackers, ts=14.0)
        sm.step(current_time=14.0, violation=None, game_frame=ready_frame)
        assert sm._advance2_ready_since != math.inf

        sm.reset()
        assert sm._advance2_ready_since == math.inf
        assert sm._advance3_ready_since == math.inf
        assert sm._advance4_ready_since == math.inf
        assert sm._last_transition_time == -math.inf

    def test_reset_preserves_construction_config(self):
        sm = GameStateMachine(
            half_duration_seconds=123.0,
            kickoff_team="blue",
            n_robots_yellow=5,
            n_robots_blue=2,
        )
        sm.reset()
        assert sm.stage_duration == 123.0
        assert sm.yellow_team.max_allowed_bots == 5
        assert sm.blue_team.max_allowed_bots == 2
        assert sm._kickoff_team_is_yellow is False


# ---------------------------------------------------------------------------
# CustomReferee integration
# ---------------------------------------------------------------------------


class TestCustomReferee:
    def test_returns_valid_referee_data(self):
        referee = CustomReferee.from_profile_name("simulation")
        frame = _frame(ball=_ball(0.0, 0.0))
        data = referee.step(frame, current_time=10.0)
        assert isinstance(data, RefereeData)
        assert data.source_identifier == "custom_referee"

    def test_simulation_goal_auto_advances_to_ball_placement_and_scores(self):
        referee = CustomReferee.from_profile_name("simulation")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        # Yellow is on the RIGHT — ball in right goal means yellow conceded, blue scored.
        frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=10.0)
        data = referee.step(frame, current_time=10.0)

        assert data.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW
        assert data.blue_team.score == 1
        assert data.yellow_team.score == 0
        assert data.next_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    def test_human_profile_no_oob(self):
        """Human profile disables out-of-bounds — ball outside must not trigger."""
        referee = CustomReferee.from_profile_name("human")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)
        frame = _frame(ball=_ball(0.0, 4.0), ts=10.0)  # ball outside field width
        data = referee.step(frame, current_time=10.0)
        assert data.referee_command == RefereeCommand.NORMAL_START

    def test_simulation_oob_exposes_status_message(self):
        referee = CustomReferee.from_profile_name("simulation")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)
        frame = _frame(ball=_ball(0.0, 4.0), ts=10.0)
        data = referee.step(frame, current_time=10.0)
        assert data.status_message == "Ball out of bounds"

    def test_human_stays_in_stop_after_goal_until_operator_advances(self):
        """Human mode keeps the game in STOP after a goal for operator control."""
        referee = CustomReferee.from_profile_name("human")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        # Score a goal (yellow on right, blue scores).
        goal_frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=10.0)
        data = referee.step(goal_frame, current_time=10.0)
        assert data.referee_command == RefereeCommand.STOP
        assert data.next_command == RefereeCommand.BALL_PLACEMENT_YELLOW

        # Still in STOP later — operator must choose the next command.
        still_stop = referee.step(_frame(ball=_ball(0.0, 0.0), ts=70.0), current_time=70.0)
        assert still_stop.referee_command == RefereeCommand.STOP

    def test_simulation_stays_in_ball_placement_after_goal_until_ball_is_placed(self):
        """Simulation mode auto-advances into BALL_PLACEMENT and waits for the ball at centre."""
        referee = CustomReferee.from_profile_name("simulation")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        goal_frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=10.0)
        referee.step(goal_frame, current_time=10.0)

        data = referee.step(_frame(ball=_ball(1.0, 0.0), ts=70.0), current_time=70.0)
        assert data.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW
        assert data.next_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    def test_simulation_oob_auto_advances_to_ball_placement_then_direct_free(self):
        referee = CustomReferee.from_profile_name("simulation")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        touch_frame = _frame(
            ball=_ball(4.4, 2.9),
            friendly_robots={0: _robot(0, 4.4, 2.9, is_friendly=True, has_ball=True)},
            my_team_is_yellow=True,
            ts=9.9,
        )
        referee.step(touch_frame, current_time=9.9)

        out_frame = _frame(ball=_ball(0.0, 3.5), my_team_is_yellow=True, ts=10.0)
        data = referee.step(out_frame, current_time=10.0)
        assert data.referee_command == RefereeCommand.BALL_PLACEMENT_BLUE
        assert data.next_command == RefereeCommand.DIRECT_FREE_BLUE

        ready_frame = _frame(
            ball=_ball(0.0, 2.9),
            friendly_robots={0: _robot(0, 1.0, 0.0, is_friendly=True)},
            enemy_robots={0: _robot(0, 0.1, 0.0, is_friendly=False)},
            my_team_is_yellow=True,
            ts=20.0,
        )
        referee.step(ready_frame, current_time=20.0)
        data = referee.step(ready_frame, current_time=23.0)
        assert data.referee_command == RefereeCommand.DIRECT_FREE_BLUE
        assert data.next_command == RefereeCommand.NORMAL_START

    def test_human_manual_direct_free_stays_in_stop_until_operator_advances(self):
        referee = CustomReferee.from_profile_name("human")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)
        referee.set_command(RefereeCommand.DIRECT_FREE_BLUE, timestamp=1.0)

        data = referee.step(_frame(ball=_ball(0.0, 0.0), my_team_is_yellow=True, ts=10.0), current_time=10.0)
        assert data.referee_command == RefereeCommand.STOP
        assert data.next_command == RefereeCommand.DIRECT_FREE_BLUE

        still_stop = referee.step(_frame(ball=_ball(0.0, 0.0), ts=40.0), current_time=40.0)
        assert still_stop.referee_command == RefereeCommand.STOP

    def test_human_manual_penalty_stays_in_stop_until_operator_advances(self):
        referee = CustomReferee.from_profile_name("human")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)
        referee.set_command(RefereeCommand.PREPARE_PENALTY_YELLOW, timestamp=1.0)

        data = referee.step(_frame(ball=_ball(0.0, 0.0), ts=10.0), current_time=10.0)
        assert data.referee_command == RefereeCommand.STOP
        assert data.next_command == RefereeCommand.PREPARE_PENALTY_YELLOW

    def test_reset_restores_score_and_command_for_episode_reuse(self):
        """A fresh episode should look exactly like a newly-constructed referee,
        without paying for a new CustomReferee instance each time (the RL
        training use case reset() exists for)."""
        referee = CustomReferee.from_profile_name("simulation")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        goal_frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=10.0)
        data = referee.step(goal_frame, current_time=10.0)
        assert data.blue_team.score == 1

        referee.reset()
        fresh_frame = _frame(ball=_ball(0.0, 0.0), ts=0.0)
        data = referee.step(fresh_frame, current_time=0.0)
        assert data.referee_command == RefereeCommand.HALT
        assert data.blue_team.score == 0
        assert data.yellow_team.score == 0

    def test_reset_clears_goal_cooldown_so_an_early_goal_in_the_new_episode_still_fires(self):
        """GoalRule.reset() deliberately keeps _last_goal_time across ordinary
        command-transition resets (mid-game cooldown). A full episode reset
        must NOT carry that timestamp into the new episode's clock, or a
        goal scored early in the new episode (small current_time) would look
        like it's still within the old episode's cooldown window and get
        suppressed."""
        referee = CustomReferee.from_profile_name("simulation")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)
        goal_frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=100.0)
        referee.step(goal_frame, current_time=100.0)

        referee.reset()
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)
        early_goal_frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=0.5)
        data = referee.step(early_goal_frame, current_time=0.5)
        assert data.blue_team.score == 1

    def test_reset_clears_keep_out_and_out_of_bounds_rule_state(self):
        referee = CustomReferee.from_profile_name("simulation")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        touch_frame = _frame(
            ball=_ball(4.4, 2.9),
            friendly_robots={0: _robot(0, 4.4, 2.9, is_friendly=True, has_ball=True)},
            my_team_is_yellow=True,
            ts=9.9,
        )
        referee.step(touch_frame, current_time=9.9)
        out_of_bounds_rule = next(r for r in referee._rules if isinstance(r, OutOfBoundsRule))
        assert out_of_bounds_rule._last_touch_was_friendly is True

        referee.reset()
        assert out_of_bounds_rule._last_touch_was_friendly is None


# ---------------------------------------------------------------------------
# Profile loader
# ---------------------------------------------------------------------------


class TestProfileLoader:
    def test_simulation_loads(self):
        profile = load_profile("simulation")
        assert profile.profile_name == "simulation"
        assert profile.rules.goal_detection.enabled is True
        assert profile.rules.keep_out.radius_meters == 0.5

    def test_human_loads(self):
        profile = load_profile("human")
        assert profile.profile_name == "human"
        assert profile.rules.out_of_bounds.enabled is False
        assert profile.game.force_start_after_goal is False
        assert profile.game.auto_advance.stop_to_next_command is False
        assert profile.game.auto_advance.prepare_penalty_to_normal is False

    def test_legacy_stop_to_prepare_kickoff_key_is_still_loaded(self, tmp_path):
        profile_path = tmp_path / "legacy_profile.yaml"
        profile_path.write_text(
            """
profile_name: "legacy"
geometry: {}
rules: {}
game:
  auto_advance:
    stop_to_prepare_kickoff: false
""".strip()
        )

        profile = load_profile(str(profile_path))
        assert profile.game.auto_advance.stop_to_next_command is False

    def test_unknown_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent_profile")
