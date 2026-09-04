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
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule

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
