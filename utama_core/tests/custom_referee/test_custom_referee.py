"""Unit tests for the CustomReferee system."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from utama_core.custom_referee.custom_referee import CustomReferee
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.profiles.profile_loader import load_profile
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

GEO = RefereeGeometry.from_standard_div_b()


def _ball(x: float, y: float, z: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, z), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))


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
    return GameStateMachine(
        half_duration_seconds=300.0,
        kickoff_team="yellow",
        n_robots_yellow=3,
        n_robots_blue=3,
        initial_command=RefereeCommand.NORMAL_START,
    )


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
        assert data.next_command == RefereeCommand.PREPARE_KICKOFF_BLUE

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

    def test_manual_set_command(self):
        sm = _state_machine()
        sm.set_command(RefereeCommand.NORMAL_START, timestamp=5.0)
        assert sm.command == RefereeCommand.NORMAL_START


# ---------------------------------------------------------------------------
# CustomReferee integration
# ---------------------------------------------------------------------------


class TestCustomReferee:
    def test_returns_valid_referee_data(self):
        referee = CustomReferee.from_profile_name("strict_ai")
        frame = _frame(ball=_ball(0.0, 0.0))
        data = referee.step(frame, current_time=10.0)
        assert isinstance(data, RefereeData)
        assert data.source_identifier == "custom_referee"

    def test_goal_triggers_stop_and_score(self):
        referee = CustomReferee.from_profile_name("strict_ai")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        # Yellow is on the RIGHT — ball in right goal means yellow conceded, blue scored.
        frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=10.0)
        data = referee.step(frame, current_time=10.0)

        assert data.referee_command == RefereeCommand.STOP
        assert data.blue_team.score == 1
        assert data.yellow_team.score == 0
        assert data.next_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    def test_arcade_profile_no_oob(self):
        """Arcade profile disables out-of-bounds — ball outside must not trigger."""
        referee = CustomReferee.from_profile_name("arcade")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)
        frame = _frame(ball=_ball(0.0, 4.0), ts=10.0)  # ball outside field width
        data = referee.step(frame, current_time=10.0)
        assert data.referee_command == RefereeCommand.NORMAL_START

    def test_arcade_auto_advances_stop_to_force_start_after_goal(self):
        """Arcade mode: after stop_duration_seconds in STOP following a goal,
        the state machine auto-advances to FORCE_START without operator input."""
        referee = CustomReferee.from_profile_name("arcade")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        # Score a goal (yellow on right, blue scores).
        goal_frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=10.0)
        data = referee.step(goal_frame, current_time=10.0)
        assert data.referee_command == RefereeCommand.STOP
        assert data.next_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

        # Still in STOP before stop_duration (arcade uses 2.0 s).
        still_stop = referee.step(_frame(ball=_ball(0.0, 0.0), ts=11.0), current_time=11.0)
        assert still_stop.referee_command == RefereeCommand.STOP

        # After stop_duration_seconds the state machine auto-issues FORCE_START.
        force_start = referee.step(_frame(ball=_ball(0.0, 0.0), ts=12.5), current_time=12.5)
        assert force_start.referee_command == RefereeCommand.FORCE_START
        assert force_start.next_command is None

    def test_strict_ai_does_not_auto_advance_after_goal(self):
        """Strict AI mode stays in STOP indefinitely — operator must issue next command."""
        referee = CustomReferee.from_profile_name("strict_ai")
        referee.set_command(RefereeCommand.NORMAL_START, timestamp=0.0)

        goal_frame = _frame(ball=_ball(5.0, 0.0), my_team_is_yellow=True, my_team_is_right=True, ts=10.0)
        referee.step(goal_frame, current_time=10.0)

        # 60 seconds later — still STOP.
        data = referee.step(_frame(ball=_ball(0.0, 0.0), ts=70.0), current_time=70.0)
        assert data.referee_command == RefereeCommand.STOP


# ---------------------------------------------------------------------------
# Profile loader
# ---------------------------------------------------------------------------


class TestProfileLoader:
    def test_strict_ai_loads(self):
        profile = load_profile("strict_ai")
        assert profile.profile_name == "strict_ai"
        assert profile.rules.goal_detection.enabled is True
        assert profile.rules.keep_out.radius_meters == 0.5

    def test_arcade_loads(self):
        profile = load_profile("arcade")
        assert profile.profile_name == "arcade"
        assert profile.rules.out_of_bounds.enabled is False
        assert profile.game.force_start_after_goal is True
        assert profile.game.stop_duration_seconds == 2.0

    def test_unknown_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent_profile")
