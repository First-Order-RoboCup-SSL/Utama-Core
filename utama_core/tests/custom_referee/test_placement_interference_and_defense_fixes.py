"""Tests for BallPlacementInterferenceRule, DefenseAreaStoppageRule, and the
Multiple Defenders sanction fix in DefenseAreaRule (SSL rulebook §8.4.1/§8.4.3).
"""

from __future__ import annotations

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.ball_placement_interference_rule import (
    BallPlacementInterferenceRule,
)
from utama_core.custom_referee.rules.defense_area_rule import DefenseAreaRule
from utama_core.custom_referee.rules.defense_area_stoppage_rule import (
    DefenseAreaStoppageRule,
)
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
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
        # The OTHER branch of DefenseAreaRule (attacker touching the ball in the
        # opponent's defense area) is still raised -- as the non-stopping foul
        # rulebook §8.4.2 makes it.
        rule = DefenseAreaRule()
        enemy = {0: _robot(0, -4.3, 0.0, is_friendly=False, has_ball=True)}
        frame = _frame(ball=_ball(-4.2, 0), enemy_robots=enemy, my_team_is_right=False, my_team_is_yellow=True)
        v = rule.check(frame, GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.is_stopping is False


class TestOutOfBoundsDefenseAreaProjection:
    """Regression for a `DIRECT_FREE_*` restart placed inside a defense area
    after an out-of-bounds ball near a goal line (found live,
    tiki_taka_plus_vs_zone_fluid, 2026-09-04): `OutOfBoundsRule`'s own
    boundary projection (`_INFIELD_OFFSET` = 0.25m) is shallower than a
    defense area's depth (`half_defense_depth` = 0.5m on the standard
    field), so a ball going out near either goal line routinely projected
    to a point still inside that defense area -- e.g. the live trace's
    (-4.25, 0.58), squarely inside the left box. `NORMAL_START` then
    immediately re-fired "too close to opponent defense area"/"attacker in
    defense area", churning into a second stoppage. Fixed by chaining the
    boundary-clamped point through `RefereeGeometry.legal_restart_position`,
    same as every other rule that derives a restart position from the
    ball's raw position already does.
    """

    def test_out_of_bounds_near_goal_line_does_not_place_inside_defense_area(self):
        rule = OutOfBoundsRule()
        # Establish last touch (friendly) with the ball in-bounds first --
        # `check()` only assigns a free kick once a touch has been observed.
        friendly = {0: _robot(0, -3.0, 0.5, is_friendly=True, has_ball=True)}
        frame_touch = _frame(
            ball=_ball(-3.0, 0.5), friendly_robots=friendly, my_team_is_right=False, my_team_is_yellow=True
        )
        assert rule.check(frame_touch, GEO, RefereeCommand.NORMAL_START) is None

        # Ball crosses the left boundary (half_length=4.5) near the live
        # trace's own y -- same shape as the real match (x slightly beyond
        # -4.5, not clamped to the goal mouth).
        frame_out = _frame(ball=_ball(-4.6, 0.576), my_team_is_right=False, my_team_is_yellow=True)
        v = rule.check(frame_out, GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.designated_position is not None
        assert not GEO.is_in_left_defense_area(*v.designated_position)
        assert not GEO.is_in_right_defense_area(*v.designated_position)
        # Left defense area's inner edge is at -3.5 (half_length - 2*depth);
        # the legal projection sits keep_dist (0.25m) plus the planner-clearance
        # buffer (0.28m) outside it -- see `legal_restart_position`'s docstring.
        px, py = v.designated_position
        assert px == pytest.approx(-2.85)
        assert py == pytest.approx(0.576)

    def test_out_of_bounds_far_from_any_defense_area_is_unaffected(self):
        # A sideline out-of-bounds well clear of either box needs no
        # defense-area projection -- only the boundary offset applies.
        rule = OutOfBoundsRule()
        friendly = {0: _robot(0, 0.0, 2.9, is_friendly=True, has_ball=True)}
        frame_touch = _frame(
            ball=_ball(0.0, 2.9), friendly_robots=friendly, my_team_is_right=False, my_team_is_yellow=True
        )
        assert rule.check(frame_touch, GEO, RefereeCommand.NORMAL_START) is None

        frame_out = _frame(ball=_ball(0.0, 3.1), my_team_is_right=False, my_team_is_yellow=True)
        v = rule.check(frame_out, GEO, RefereeCommand.NORMAL_START)
        assert v is not None
        assert v.designated_position == pytest.approx((0.0, 2.75))  # half_width (3.0) - 0.25
