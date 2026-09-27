"""Static legality sweep for `RefereeGeometry.legal_restart_position`.

This is the "cheap self-test" from the 2026-09-04 session's observability
follow-up: the out-of-bounds/defense-area placement bug found live in
`tiki_taka_plus_vs_zone_fluid` (see `OutOfBoundsRule._nearest_infield_point`'s
docstring) was a pure property of this one shared helper -- for ANY input
point, the output must be in-field and clear of both defense areas -- and
that property is checkable without a live match or replay data at all. A
dense grid sweep across the whole field (plus points just outside it, since
every one of the 7 callers documented in `legal_restart_position`'s own
docstring feeds it a *raw* ball position that can itself be out of bounds)
is exactly the static check that would have caught the original bug before
it ever needed a human to spot it on a dashboard replay.

No `hypothesis` dependency in this repo (checked 2026-09-04) -- a fixed
grid at sub-metre resolution over the field's full extent (plus a margin
for out-of-field inputs) gives the same exhaustive-enough coverage a
property test would, without adding one.
"""

from __future__ import annotations

import itertools

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.out_of_bounds_rule import (
    _CORNER_INFIELD_OFFSET,
    OutOfBoundsRule,
)

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)

# Sweep a margin beyond the field boundary too: every real caller
# (ball_speed_rule, defense_area_rule, double_touch_rule, keeper_held_ball_rule,
# excessive_dribbling_rule, pushing_rule) can feed this a ball position that is
# itself out of bounds (the ball just left the field the same tick a rule
# fires), so restricting the sweep to in-field inputs would miss exactly the
# class of input that caused the original bug.
_MARGIN = 1.0
_STEP = 0.2


def _grid(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


_XS = _grid(-GEO.half_length - _MARGIN, GEO.half_length + _MARGIN, _STEP)
_YS = _grid(-GEO.half_width - _MARGIN, GEO.half_width + _MARGIN, _STEP)


def _in_either_defense_area(x: float, y: float) -> bool:
    return GEO.is_in_left_defense_area(x, y) or GEO.is_in_right_defense_area(x, y)


class TestLegalRestartPositionIsAlwaysLegal:
    """`legal_restart_position(x, y, keep_dist)` must never return a point
    inside either defense area, for any (x, y) input -- this is the exact
    contract every one of its 7 callers relies on (see its own docstring).
    """

    @pytest.mark.parametrize("x,y", list(itertools.product(_XS, _YS)))
    def test_output_clears_both_defense_areas(self, x: float, y: float) -> None:
        px, py = GEO.legal_restart_position(x, y, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
        assert not GEO.is_in_left_defense_area(px, py), f"input=({x},{y}) -> output=({px},{py}) still in left DA"
        assert not GEO.is_in_right_defense_area(px, py), f"input=({x},{y}) -> output=({px},{py}) still in right DA"

    @pytest.mark.parametrize("x,y", list(itertools.product(_XS, _YS)))
    def test_output_preserves_y_and_only_moves_x(self, x: float, y: float) -> None:
        """The helper's own docstring asserts y never needs clamping -- only
        x moves. Pin that invariant directly so a future edit that breaks it
        (e.g. a non-standard field where a defense area isn't full-width)
        fails loudly here instead of silently shipping a diagonal-only fix.
        """
        px, py = GEO.legal_restart_position(x, y, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
        assert py == y


class TestLegalRestartPositionClearsPlannerObstacle:
    """`legal_restart_position` must not just land outside a defense area --
    it must clear `FastPathPlanner`'s own obstacle-clearance ring around that
    same edge, or the delivering/kicking robot's path-planning carrot never
    reaches the placed point (`sanitize_target` pushes the carrot further
    away from a target sitting inside its clearance ring, but the referee
    state machine's ball-placement-done check still measures against the
    *original*, un-pushed point).

    Traced live 2026-09-05 (`counter_flow_vs_tiki_taka_RK`, BALL_PLACEMENT_YELLOW,
    72.7s RESTART_STALL): `designated_position=(-3.25, 0.587)` sat exactly
    `OPPONENT_DEFENSE_AREA_KEEP_DISTANCE` (0.25m) from the defense area's raw
    edge -- legal by `is_in_left_defense_area`'s strict-inside test, but
    exactly on `FastPathPlanner`'s obstacle line for that same edge (it uses
    the identical margin as its base clearance distance), so the carrier's
    carrot converged ~0.37m short of it and the restart never auto-advanced
    for the rest of the match. Confirmed via `git stash` that this test fails
    against the pre-fix `legal_restart_position` (which clamped to exactly
    `keep_dist`, not `keep_dist + _PLANNER_CLEARANCE_BUFFER_M`).
    """

    # `fastpathplanningconfig.OBSTACLE_CLEARANCE` (ROBOT_DIAMETER * 1.5 = 0.27m
    # at full, non-crowded clearance) -- not imported directly to keep this
    # referee-layer test free of a motion-planning-layer dependency, per
    # `legal_restart_position`'s own docstring rationale for the same
    # buffer. Kept in sync by the pytest.approx tolerance below being wide
    # enough to catch either module drifting relative to the other.
    _PLANNER_FULL_CLEARANCE_M = 0.27

    @pytest.mark.parametrize("x,y", list(itertools.product(_XS, _YS)))
    def test_output_clears_planner_obstacle_margin(self, x: float, y: float) -> None:
        px, py = GEO.legal_restart_position(x, y, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
        if abs(py) > GEO.half_defense_width:
            return  # legal_restart_position only clamps x when y is within box width
        left_inner_x = -GEO.half_length + 2.0 * GEO.half_defense_depth
        right_inner_x = GEO.half_length - 2.0 * GEO.half_defense_depth
        dist_from_left_edge = px - left_inner_x
        dist_from_right_edge = right_inner_x - px
        # The point must clear at least one side's full planner-obstacle
        # margin (whichever edge it was actually clamped against, or its
        # original position if neither needed clamping and it was already
        # this far out) -- not just the bare `keep_dist`.
        assert dist_from_left_edge >= self._PLANNER_FULL_CLEARANCE_M + OPPONENT_DEFENSE_AREA_KEEP_DISTANCE - 1e-9 or (
            x <= left_inner_x
        ), f"input=({x},{y}) -> output=({px},{py}) only {dist_from_left_edge:.3f}m from left DA edge"
        assert dist_from_right_edge >= self._PLANNER_FULL_CLEARANCE_M + OPPONENT_DEFENSE_AREA_KEEP_DISTANCE - 1e-9 or (
            x >= right_inner_x
        ), f"input=({x},{y}) -> output=({px},{py}) only {dist_from_right_edge:.3f}m from right DA edge"


class TestOutOfBoundsPlacementIsAlwaysLegal:
    """End-to-end version of the same property through the actual caller
    that had the bug: any ball position outside the field must resolve to a
    designated_position that is both in-field and clear of both defense
    areas. Restricted to points strictly outside the field (an in-field
    point never reaches `OutOfBoundsRule._nearest_infield_point` in
    practice -- `check()` only calls it once `is_in_field` has already
    failed), plus enough margin to cover realistic overshoot.
    """

    _OOB_XS = [x for x in _XS if abs(x) > GEO.half_length]
    _OOB_YS_ANY_X = _YS  # a ball can go out over either the goal line or the sideline

    @pytest.mark.parametrize("x,y", list(itertools.product(_OOB_XS, _OOB_YS_ANY_X)))
    def test_placement_in_field_and_clear_of_defense_areas(self, x: float, y: float) -> None:
        px, py = OutOfBoundsRule._nearest_infield_point(x, y, GEO)
        assert GEO.is_in_field(px, py), f"input=({x},{y}) -> placement=({px},{py}) not in field"
        assert not _in_either_defense_area(px, py), f"input=({x},{y}) -> placement=({px},{py}) inside a defense area"

    @pytest.mark.parametrize("y", [y for y in _YS if abs(y) > GEO.half_width])
    def test_sideline_out_placement_in_field_and_clear(self, y: float) -> None:
        """Same property for a ball going out over a sideline (x in-field,
        |y| > half_width) -- the family of input this rule most commonly
        sees, kept as its own case for a readable failure if it regresses.
        """
        for x in (-2.0, 0.0, 2.0):
            px, py = OutOfBoundsRule._nearest_infield_point(x, y, GEO)
            assert GEO.is_in_field(px, py)
            assert not _in_either_defense_area(px, py)


# The corner double-boundary deadlock: an exit near a corner (close to BOTH
# the goal line and the sideline at once) used to get only _INFIELD_OFFSET
# (0.25m) of clearance from EACH edge independently -- as little as 0.08m
# observed live -- letting an ordinary post-restart drift send the ball back
# out one of the two nearby lines and re-trigger the same restart, looping in
# the corner for most or all of a match (found live 2026-09-04 across 4 real
# tournament matches, e.g. counter_flow_vs_counter_press: 58 out-of-bounds
# events in one 65s match). See _nearest_infield_point's own docstring for
# the full mechanism and the fix.
_CORNER_MARGIN_XS = [x for x in _XS if GEO.half_length - abs(x) < 1.0]  # within 1m of the goal line, either side
_CORNER_MARGIN_YS_OOB = [y for y in _YS if abs(y) > GEO.half_width]  # any sideline-out y


class TestOutOfBoundsCornerPlacementHasRealClearance:
    """A corner-region out-of-bounds placement must sit at least
    `_CORNER_INFIELD_OFFSET` from BOTH the goal line and the sideline at
    once, not just individually legal per-axis -- the property the original
    corner-deadlock bug violated."""

    @pytest.mark.parametrize("x,y", list(itertools.product(_CORNER_MARGIN_XS, _CORNER_MARGIN_YS_OOB)))
    def test_corner_exit_gets_real_clearance_from_both_edges(self, x: float, y: float) -> None:
        px, py = OutOfBoundsRule._nearest_infield_point(x, y, GEO)
        dist_goal_line = GEO.half_length - abs(px)
        dist_sideline = GEO.half_width - abs(py)
        # Both edges must clear the deeper corner offset whenever the RAW
        # exit was within it of both edges -- mirrors _nearest_infield_point's
        # own near_corner check, so this fails loudly if that detection or
        # the offset it applies regresses.
        raw_near_goal_line = GEO.half_length - abs(x) < _CORNER_INFIELD_OFFSET
        raw_near_sideline = GEO.half_width - abs(y) < _CORNER_INFIELD_OFFSET
        if raw_near_goal_line and raw_near_sideline:
            assert (
                dist_goal_line >= _CORNER_INFIELD_OFFSET - 1e-9
            ), f"input=({x},{y}) -> placement=({px},{py}) only {dist_goal_line:.3f}m from goal line"
            assert (
                dist_sideline >= _CORNER_INFIELD_OFFSET - 1e-9
            ), f"input=({x},{y}) -> placement=({px},{py}) only {dist_sideline:.3f}m from sideline"


class TestOutOfBoundsCornerPlacementRealTraces:
    """Exact ball positions traced live from the fresh tournament run that
    exposed this bug (`replays/tournament_20260904_221937/`, current HEAD).
    Each of these previously placed a restart 0.08-0.30m from both nearby
    edges at once; every one must now clear `_CORNER_INFIELD_OFFSET` on
    both axes."""

    # (source match, exit x, exit y)
    _TRACED_EXITS = [
        ("counter_flow_vs_counter_press", 4.501422619095742, 2.9200460783266107),
        ("counter_flow_vs_counter_press", 4.308722938798151, 3.005813148990557),
        ("counter_flow_vs_counter_press", 4.362237866288889, 3.005589533158603),
        ("counter_flow_vs_counter_press", 4.507241984858946, 2.69573238652152),
        ("counter_flow_vs_counter_press", 4.501868255855371, 2.835103084601348),
        ("counter_flow_vs_counter_press", 4.2529424410160885, 3.000305091393659),
        ("overload_press_vs_tiki_taka_plus", -4.524, -2.590),
        ("overload_press_vs_tiki_taka_plus", -4.505, -2.735),
        ("clear_danger_vs_counter_press", 4.502, 2.966),
        ("clear_danger_vs_counter_press", 4.251, 3.007),
    ]

    @pytest.mark.parametrize("source,x,y", _TRACED_EXITS)
    def test_traced_corner_exit_now_has_real_clearance(self, source: str, x: float, y: float) -> None:
        px, py = OutOfBoundsRule._nearest_infield_point(x, y, GEO)
        dist_goal_line = GEO.half_length - abs(px)
        dist_sideline = GEO.half_width - abs(py)
        assert (
            dist_goal_line >= _CORNER_INFIELD_OFFSET - 1e-9
        ), f"{source}: exit=({x},{y}) -> placement=({px},{py}) only {dist_goal_line:.3f}m from goal line"
        assert (
            dist_sideline >= _CORNER_INFIELD_OFFSET - 1e-9
        ), f"{source}: exit=({x},{y}) -> placement=({px},{py}) only {dist_sideline:.3f}m from sideline"
        assert GEO.is_in_field(px, py)
        assert not _in_either_defense_area(px, py)


@pytest.mark.parametrize("x,y", [(-4.25, -1.004), (-4.25, 1.02), (4.25, -1.1), (3.9, 1.2)])
def test_a_restart_beside_the_box_side_edge_clears_the_planner_margin(x: float, y: float) -> None:
    """A restart level with the box only needed pushing out if it was inside the box's
    width, so one just past the side edge stayed where it was: a free kick at
    (-4.25, -1.00), the box corner, left the kicker parked outside the planner's
    keep-out ring 0.49 m away for the rest of the match (three matches,
    tournament_20260927_223257). The keep distance applies at the side edge too."""
    px, py = GEO.legal_restart_position(x, y, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
    inner_x = GEO.half_length - 2.0 * GEO.half_defense_depth
    beyond_front = inner_x - abs(px)
    beyond_side = abs(py) - GEO.half_defense_width
    clearance = max(beyond_front, beyond_side)
    assert (
        clearance
        >= OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
        + TestLegalRestartPositionClearsPlannerObstacle._PLANNER_FULL_CLEARANCE_M
        - 1e-9
    )


@pytest.mark.parametrize("x,y", [(-4.25, -1.54), (4.25, 1.6), (-3.1, 1.2)])
def test_a_restart_near_the_box_leaves_room_for_the_kickers_approach(x: float, y: float) -> None:
    """The kicker stands `DirectFreeOursStep._APPROACH_OFFSET` behind the ball, which is
    towards the box whenever the kick points away from it; that spot has to clear the
    planner's ring too. A free kick at (-4.25, -1.54), 0.54 m from the side edge, held
    its kicker 1.3 m away for 20 s (tournament_20260927_230330)."""
    from utama_core.custom_referee.actions import DirectFreeOursStep

    px, py = GEO.legal_restart_position(x, y, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
    inner_x = GEO.half_length - 2.0 * GEO.half_defense_depth
    clearance = max(inner_x - abs(px), abs(py) - GEO.half_defense_width)
    need = (
        OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
        + TestLegalRestartPositionClearsPlannerObstacle._PLANNER_FULL_CLEARANCE_M
        + DirectFreeOursStep._APPROACH_OFFSET
    )
    assert clearance >= need - 1e-9
