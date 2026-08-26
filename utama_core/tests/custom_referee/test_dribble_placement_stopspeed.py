"""Unit tests for KeeperHeldBallRule, ExcessiveDribblingRule, RobotStopSpeedRule
(SSL rulebook §8.4.1/§8.4.3)."""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.excessive_dribbling_rule import (
    ExcessiveDribblingRule,
)
from utama_core.custom_referee.rules.keeper_held_ball_rule import KeeperHeldBallRule
from utama_core.custom_referee.rules.robot_stop_speed_rule import RobotStopSpeedRule
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)


def _ball(x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, 0.0), v=Vector3D(vx, vy, 0.0), a=Vector3D(0, 0, 0))


def _robot(
    robot_id: int, x: float, y: float, is_friendly: bool, has_ball: bool = False, vx: float = 0.0, vy: float = 0.0
) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(vx, vy),
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


# ---------------------------------------------------------------------------
# KeeperHeldBallRule
# ---------------------------------------------------------------------------


class TestKeeperHeldBallRule:
    def _in_own_area_frame(self, ts: float) -> GameFrame:
        # my_team_is_yellow=True, my_team_is_right=False -> yellow defends
        # the LEFT area (x <= -3.5). Ball resting deep in yellow's own box.
        return _frame(ball=_ball(-4.0, 0.0), my_team_is_yellow=True, my_team_is_right=False, ts=ts)

    def test_no_violation_under_threshold(self):
        rule = KeeperHeldBallRule(max_hold_seconds=10.0)
        v = None
        for ts in [0.0, 3.0, 6.0, 9.0]:
            v = rule.check(self._in_own_area_frame(ts), GEO, RefereeCommand.NORMAL_START)
        assert v is None

    def test_violation_past_threshold(self):
        rule = KeeperHeldBallRule(max_hold_seconds=10.0)
        v = None
        for ts in [0.0, 5.0, 10.5]:
            v = rule.check(self._in_own_area_frame(ts), GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.rule_name == "keeper_held_ball"
        # Yellow's own area held too long -> free kick to blue, yellow charged.
        assert v.next_command == RefereeCommand.DIRECT_FREE_BLUE
        assert v.offending_teams == (True,)

    def test_clock_resets_when_ball_leaves_area(self):
        rule = KeeperHeldBallRule(max_hold_seconds=10.0)
        rule.check(self._in_own_area_frame(0.0), GEO, RefereeCommand.NORMAL_START)
        rule.check(self._in_own_area_frame(8.0), GEO, RefereeCommand.NORMAL_START)
        # Ball leaves the area.
        out_frame = _frame(ball=_ball(0.0, 0.0), my_team_is_yellow=True, my_team_is_right=False, ts=9.0)
        v = rule.check(out_frame, GEO, RefereeCommand.NORMAL_START)
        assert v is None
        # Re-enters; clock must have restarted, not resumed from 8s.
        v = rule.check(self._in_own_area_frame(10.0), GEO, RefereeCommand.NORMAL_START)
        assert v is None


# ---------------------------------------------------------------------------
# ExcessiveDribblingRule
# ---------------------------------------------------------------------------


class TestExcessiveDribblingRule:
    def test_no_violation_under_1m(self):
        rule = ExcessiveDribblingRule(max_dribble_meters=1.0)
        v = None
        for x in [0.0, 0.3, 0.6, 0.9]:
            robot = _robot(0, x, 0.0, is_friendly=True, has_ball=True)
            v = rule.check(_frame(ball=_ball(x, 0.0), friendly_robots={0: robot}), GEO, RefereeCommand.NORMAL_START)
        assert v is None

    def test_violation_past_1m(self):
        rule = ExcessiveDribblingRule(max_dribble_meters=1.0)
        v = None
        for x in [0.0, 0.5, 1.2]:
            robot = _robot(0, x, 0.0, is_friendly=True, has_ball=True)
            v = rule.check(
                _frame(ball=_ball(x, 0.0), friendly_robots={0: robot}, my_team_is_yellow=True),
                GEO,
                RefereeCommand.NORMAL_START,
            )
        assert v is not None
        assert v.rule_name == "excessive_dribbling"
        assert v.next_command == RefereeCommand.DIRECT_FREE_BLUE
        assert v.offending_teams == (True,)

    def test_kicking_ahead_resets_origin(self):
        """Two separate sub-1m dribbles (with an observable separation
        between them) must not accumulate into a false foul, even though
        their combined displacement exceeds 1m — the rulebook's explicit
        "kicking the ball ahead of it" exemption."""
        rule = ExcessiveDribblingRule(max_dribble_meters=1.0)
        violations = []

        # First dribble: 0.0 -> 0.8 (has_ball throughout).
        for x in [0.0, 0.4, 0.8]:
            robot = _robot(0, x, 0.0, is_friendly=True, has_ball=True)
            violations.append(
                rule.check(_frame(ball=_ball(x, 0.0), friendly_robots={0: robot}), GEO, RefereeCommand.NORMAL_START)
            )

        # Observable separation: has_ball drops for a tick while the ball
        # continues past the robot (a kick).
        robot_no_ball = _robot(0, 0.8, 0.0, is_friendly=True, has_ball=False)
        violations.append(
            rule.check(
                _frame(ball=_ball(1.0, 0.0), friendly_robots={0: robot_no_ball}), GEO, RefereeCommand.NORMAL_START
            )
        )

        # Second dribble starts fresh at x=1.0, only travels to 1.7 (0.7m
        # from its own origin) even though 1.7 - 0.0 = 1.7m > 1.0m overall.
        for x in [1.0, 1.3, 1.7]:
            robot = _robot(0, x, 0.0, is_friendly=True, has_ball=True)
            violations.append(
                rule.check(_frame(ball=_ball(x, 0.0), friendly_robots={0: robot}), GEO, RefereeCommand.NORMAL_START)
            )

        assert all(v is None for v in violations)


# ---------------------------------------------------------------------------
# RobotStopSpeedRule
# ---------------------------------------------------------------------------


class TestRobotStopSpeedRule:
    def test_no_violation_during_grace_period(self):
        rule = RobotStopSpeedRule(max_speed_mps=1.5, grace_seconds=2.0)
        fast_robot = _robot(0, 0.0, 0.0, is_friendly=True, vx=3.0, vy=0.0)
        v = None
        for ts in [0.0, 0.5, 1.0, 1.9]:
            v = rule.check(
                _frame(ball=_ball(2.0, 2.0), friendly_robots={0: fast_robot}, ts=ts), GEO, RefereeCommand.STOP
            )
        assert v is None

    def test_violation_after_grace_period(self):
        rule = RobotStopSpeedRule(max_speed_mps=1.5, grace_seconds=2.0)
        fast_robot = _robot(0, 0.0, 0.0, is_friendly=True, vx=3.0, vy=0.0)
        rule.check(_frame(ball=_ball(2.0, 2.0), friendly_robots={0: fast_robot}, ts=0.0), GEO, RefereeCommand.STOP)
        v = rule.check(_frame(ball=_ball(2.0, 2.0), friendly_robots={0: fast_robot}, ts=2.5), GEO, RefereeCommand.STOP)
        assert v is not None
        assert v.rule_name == "robot_stop_speed"
        assert v.is_stopping is False
        assert v.offending_teams == (True,)
        # Non-stopping report: command must not be perturbed.
        assert v.suggested_command == RefereeCommand.STOP

    def test_not_double_counted_within_one_stoppage(self):
        rule = RobotStopSpeedRule(max_speed_mps=1.5, grace_seconds=2.0)
        fast_robot = _robot(0, 0.0, 0.0, is_friendly=True, vx=3.0, vy=0.0)
        rule.check(_frame(ball=_ball(2.0, 2.0), friendly_robots={0: fast_robot}, ts=0.0), GEO, RefereeCommand.STOP)
        v1 = rule.check(_frame(ball=_ball(2.0, 2.0), friendly_robots={0: fast_robot}, ts=2.5), GEO, RefereeCommand.STOP)
        v2 = rule.check(_frame(ball=_ball(2.0, 2.0), friendly_robots={0: fast_robot}, ts=3.0), GEO, RefereeCommand.STOP)
        assert v1 is not None
        assert v2 is None

    def test_does_not_fire_during_ball_placement(self):
        rule = RobotStopSpeedRule(max_speed_mps=1.5, grace_seconds=2.0)
        fast_robot = _robot(0, 0.0, 0.0, is_friendly=True, vx=3.0, vy=0.0)
        v = None
        for ts in [0.0, 3.0, 6.0]:
            v = rule.check(
                _frame(ball=_ball(2.0, 2.0), friendly_robots={0: fast_robot}, ts=ts),
                GEO,
                RefereeCommand.BALL_PLACEMENT_YELLOW,
            )
        assert v is None

    def test_exempt_while_still_inside_keep_out_zone_past_grace_period(self):
        """A robot that entered STOP already inside BALL_KEEP_OUT_DISTANCE (0.8m)
        of the ball is being actively driven out by RefereeOverride's
        `_clear_to_legal_positions` at full motion-controller speed. Fouling it
        for that — just because the 2s grace clock expired before it physically
        cleared 0.8m — would penalize the robot for complying with the
        referee's own override. See docs/testing_gaps.md / the referee-override
        restart-safety audit (2026-08-26).
        """
        rule = RobotStopSpeedRule(max_speed_mps=1.5, grace_seconds=2.0)
        # Robot sits 0.3m from the ball (inside the 0.8m keep-out zone) and is
        # moving fast, as it would be while RefereeOverride drives it out.
        robot_still_clearing = _robot(0, 0.3, 0.0, is_friendly=True, vx=3.0, vy=0.0)
        ball = _ball(0.0, 0.0)
        v = None
        for ts in [0.0, 1.0, 2.5, 4.0]:
            v = rule.check(
                _frame(ball=ball, friendly_robots={0: robot_still_clearing}, ts=ts), GEO, RefereeCommand.STOP
            )
        assert v is None

    def test_fires_once_robot_clears_keep_out_zone_and_still_speeds(self):
        """Once a robot is outside the keep-out zone (i.e. has had the chance to
        comply), the ordinary grace-period/speed check applies as before.
        """
        rule = RobotStopSpeedRule(max_speed_mps=1.5, grace_seconds=2.0)
        # 2.0m from the ball: well outside the 0.8m keep-out radius.
        robot_clear_of_zone = _robot(0, 2.0, 0.0, is_friendly=True, vx=3.0, vy=0.0)
        ball = _ball(0.0, 0.0)
        rule.check(_frame(ball=ball, friendly_robots={0: robot_clear_of_zone}, ts=0.0), GEO, RefereeCommand.STOP)
        v = rule.check(_frame(ball=ball, friendly_robots={0: robot_clear_of_zone}, ts=2.5), GEO, RefereeCommand.STOP)
        assert v is not None
        assert v.rule_name == "robot_stop_speed"
