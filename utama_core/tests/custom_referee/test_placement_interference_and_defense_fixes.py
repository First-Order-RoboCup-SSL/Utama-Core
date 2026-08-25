"""Tests for BallPlacementInterferenceRule, DefenseAreaStoppageRule, and the
Multiple Defenders sanction fix in DefenseAreaRule (SSL rulebook §8.4.1/§8.4.3).
"""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.ball_placement_interference_rule import (
    BallPlacementInterferenceRule,
)
from utama_core.custom_referee.rules.defense_area_rule import DefenseAreaRule
from utama_core.custom_referee.rules.defense_area_stoppage_rule import (
    DefenseAreaStoppageRule,
)
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)


def _ball(x: float, y: float) -> Ball:
    return Ball(p=Vector3D(x, y, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))


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


# ---------------------------------------------------------------------------
# BallPlacementInterferenceRule
# ---------------------------------------------------------------------------


class TestBallPlacementInterferenceRule:
    def test_no_violation_when_clear_of_stadium_zone(self):
        rule = BallPlacementInterferenceRule(stadium_radius_meters=0.5, grace_seconds=2.0)
        # Ball at origin, placement target at (2, 0); enemy robot far off the line.
        enemy = {0: _robot(0, 1.0, 3.0, is_friendly=False)}
        for t in range(10):
            frame = _frame(ball=_ball(0.0, 0.0), enemy_robots=enemy, ts=float(t))
            v = rule.check(frame, GEO, RefereeCommand.BALL_PLACEMENT_YELLOW, designated_position=(2.0, 0.0))
            assert v is None

    def test_no_violation_under_grace_period(self):
        rule = BallPlacementInterferenceRule(stadium_radius_meters=0.5, grace_seconds=2.0)
        # Enemy robot sitting right on the ball-to-target line.
        enemy = {0: _robot(0, 1.0, 0.0, is_friendly=False)}
        for t in [10.0, 10.5, 11.0, 11.9]:
            frame = _frame(ball=_ball(0.0, 0.0), enemy_robots=enemy, ts=t)
            v = rule.check(frame, GEO, RefereeCommand.BALL_PLACEMENT_YELLOW, designated_position=(2.0, 0.0))
            assert v is None

    def test_violation_past_grace_period_charges_non_placing_team(self):
        rule = BallPlacementInterferenceRule(stadium_radius_meters=0.5, grace_seconds=2.0)
        enemy = {0: _robot(0, 1.0, 0.0, is_friendly=False)}
        v = None
        for t in [10.0, 11.0, 12.1]:
            frame = _frame(ball=_ball(0.0, 0.0), enemy_robots=enemy, ts=t)
            v = rule.check(frame, GEO, RefereeCommand.BALL_PLACEMENT_YELLOW, designated_position=(2.0, 0.0))
        assert v is not None
        assert v.rule_name == "ball_placement_interference"
        # Yellow is placing (friendly, per my_team_is_yellow=True default) → enemy (blue) is non-placing/offending.
        assert v.offending_teams == (False,)
        assert v.counts_toward_foul_counter is True
        assert v.is_stopping is False

    def test_only_first_interference_per_phase_counts_toward_foul_counter(self):
        rule = BallPlacementInterferenceRule(stadium_radius_meters=0.5, grace_seconds=2.0)
        enemy = {0: _robot(0, 1.0, 0.0, is_friendly=False)}
        violations = []
        for t in [10.0, 11.0, 12.1, 13.0, 14.1]:
            frame = _frame(ball=_ball(0.0, 0.0), enemy_robots=enemy, ts=t)
            v = rule.check(frame, GEO, RefereeCommand.BALL_PLACEMENT_YELLOW, designated_position=(2.0, 0.0))
            if v is not None:
                violations.append(v)
        assert len(violations) == 2  # fires again after re-arming, but only the first counts
        assert violations[0].counts_toward_foul_counter is True
        assert violations[1].counts_toward_foul_counter is False

    def test_no_violation_outside_ball_placement_command(self):
        rule = BallPlacementInterferenceRule(stadium_radius_meters=0.5, grace_seconds=2.0)
        enemy = {0: _robot(0, 1.0, 0.0, is_friendly=False)}
        for t in [10.0, 11.0, 12.1]:
            frame = _frame(ball=_ball(0.0, 0.0), enemy_robots=enemy, ts=t)
            v = rule.check(frame, GEO, RefereeCommand.NORMAL_START, designated_position=(2.0, 0.0))
            assert v is None


# ---------------------------------------------------------------------------
# DefenseAreaStoppageRule
# ---------------------------------------------------------------------------


class TestDefenseAreaStoppageRule:
    def _yellow_near_blue_defense(self, gap: float) -> dict:
        # my_team_is_yellow=True, my_team_is_right=False (used by _frame's
        # defaults below) -> yellow_is_right = (my_team_is_right ==
        # my_team_is_yellow) = (False == True) = False, so yellow defends
        # LEFT and blue (yellow's opponent) defends RIGHT. Place a yellow
        # (friendly) robot `gap` metres outside the RIGHT defense area's
        # boundary (right defense rect starts at x = half_length -
        # 2*half_defense_depth) — pass gap < min_distance to violate.
        rect_x = STANDARD_FIELD_DIMS.full_field_half_length - 2 * STANDARD_FIELD_DIMS.half_defense_area_depth
        return {0: _robot(0, rect_x - gap, 0.0, is_friendly=True)}

    def test_no_violation_when_clear(self):
        rule = DefenseAreaStoppageRule(min_distance_meters=0.2, grace_seconds=2.0)
        friendly = {0: _robot(0, 0.0, 0.0, is_friendly=True)}
        for t in range(10):
            frame = _frame(ball=_ball(0, 0), friendly_robots=friendly, my_team_is_yellow=True, ts=float(t))
            v = rule.check(frame, GEO, RefereeCommand.STOP)
            assert v is None

    def test_no_violation_within_grace_period(self):
        rule = DefenseAreaStoppageRule(min_distance_meters=0.2, grace_seconds=2.0)
        friendly = self._yellow_near_blue_defense(gap=0.05)
        for t in [10.0, 11.0, 11.9]:
            frame = _frame(ball=_ball(0, 0), friendly_robots=friendly, my_team_is_yellow=True, ts=t)
            v = rule.check(frame, GEO, RefereeCommand.STOP)
            assert v is None

    def test_first_foul_past_grace_stops_regularly(self):
        rule = DefenseAreaStoppageRule(min_distance_meters=0.2, grace_seconds=2.0)
        friendly = self._yellow_near_blue_defense(gap=0.05)
        v = None
        for t in [10.0, 11.0, 12.1]:
            frame = _frame(ball=_ball(0, 0), friendly_robots=friendly, my_team_is_yellow=True, ts=t)
            v = rule.check(frame, GEO, RefereeCommand.STOP)
        assert v is not None
        assert v.offending_teams == (True,)
        assert v.is_stopping is False  # "the game is still stopped regularly" — no forced HALT yet

    def test_second_foul_by_same_team_halts(self):
        rule = DefenseAreaStoppageRule(min_distance_meters=0.2, grace_seconds=2.0)
        friendly = self._yellow_near_blue_defense(gap=0.05)
        v = None
        # First foul at t=12.1 (grace elapsed from t=10.0), second foul after
        # the grace period re-arms from t=12.1.
        for t in [10.0, 11.0, 12.1, 13.1, 14.2]:
            frame = _frame(ball=_ball(0, 0), friendly_robots=friendly, my_team_is_yellow=True, ts=t)
            v = rule.check(frame, GEO, RefereeCommand.STOP)
        assert v is not None
        assert v.suggested_command == RefereeCommand.HALT
        assert v.offending_teams == (True,)


# ---------------------------------------------------------------------------
# DefenseAreaRule — Multiple Defenders sanction fix
# ---------------------------------------------------------------------------


class TestMultipleDefendersSanctionFix:
    def test_no_violation_on_occupancy_alone_without_ball_touch(self):
        rule = DefenseAreaRule(max_defenders=1)
        friendly = {
            0: _robot(0, -4.3, 0.0, is_friendly=True),
            1: _robot(1, -4.3, 0.5, is_friendly=True),
        }
        frame = _frame(ball=_ball(0, 0), friendly_robots=friendly, my_team_is_right=False, my_team_is_yellow=True)
        v = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert v is None  # occupancy alone is no longer sufficient — needs ball touch

    def test_penalty_kick_when_extra_defender_touches_ball(self):
        rule = DefenseAreaRule(max_defenders=1)
        friendly = {
            0: _robot(0, -4.3, 0.0, is_friendly=True),
            1: _robot(1, -4.3, 0.5, is_friendly=True, has_ball=True),
        }
        frame = _frame(ball=_ball(-4.3, 0.5), friendly_robots=friendly, my_team_is_right=False, my_team_is_yellow=True)
        v = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.next_command == RefereeCommand.PREPARE_PENALTY_BLUE
        assert v.counts_toward_foul_counter is False

    def test_attacker_infringement_branch_unaffected(self):
        # The OTHER branch of DefenseAreaRule (attacker in opponent's own
        # defense area) must be untouched by this fix.
        rule = DefenseAreaRule()
        enemy = {0: _robot(0, -4.3, 0.0, is_friendly=False)}
        frame = _frame(ball=_ball(0, 0), enemy_robots=enemy, my_team_is_right=False, my_team_is_yellow=True)
        v = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.next_command == RefereeCommand.DIRECT_FREE_YELLOW
