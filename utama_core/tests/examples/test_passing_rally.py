"""`examples/passing_rally.py`: `pass_receive.py`'s pair passing along the field in two lanes, to one
end and back. Every rule here is a pure function of the world (and the way the rally heads), as both
sides read it."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.shared.pass_and_score_geometry import reset_possession_state
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import pass_receive as pr
    import passing_rally as ex
finally:
    sys.path.remove(str(EXAMPLES))


@pytest.fixture(autouse=True)
def _fresh_possession_state():
    reset_possession_state()
    yield
    reset_possession_state()


def _game(me: dict, partner: dict, ball):
    return build_game(
        {0: make_robot(0, is_friendly=True, **me)},
        {0: make_robot(0, is_friendly=False, **partner)},
        ball,
        my_team_is_yellow=True,
        my_team_is_right=False,
        field_dims=pr.FIELD,
        defense_area_rules=False,
    )


@pytest.mark.parametrize(
    "sign, ball_x, then",
    [
        (None, -0.8, 1.0),  # a new rally heads for the end further from the ball
        (None, 0.8, -1.0),
        (1.0, 0.0, 1.0),
        (1.0, "short", 1.0),  # just short of the turn
        (1.0, "past", -1.0),  # within TURN_M of the +x end line: back
        (-1.0, "past", -1.0),  # heading away from the end it is near: keeps going
        (-1.0, "-past", 1.0),
    ],
)
def test_the_rally_turns_round_near_the_end_it_heads_for(sign, ball_x, then):
    turn_x = _game({"x": 0.0}, {"x": 0.5}, make_ball()).field.half_length - ex.TURN_M
    x = {"short": turn_x - 0.01, "past": turn_x + 0.01, "-past": -turn_x - 0.01}.get(ball_x, ball_x)
    g = _game({"x": 0.0, "y": 0.5}, {"x": 0.0, "y": -0.5}, make_ball(x=x))
    assert ex.rally_direction(g, sign) == then


def test_the_rally_goes_end_to_end_and_back_for_ever():
    """The ball carried along by each pass: the way the rally heads flips at each end, every time."""
    sign, flips, x = None, [], -1.0
    for _ in range(40):
        g = _game({"x": x - 0.1}, {"x": 0.0, "y": 0.5}, make_ball(x=x, y=0.2))
        new = ex.rally_direction(g, sign)
        if sign is not None and new != sign:
            flips.append(x)
        sign = new
        x = ex.upfield_spot(g, Vector2D(0.0, 0.5), sign).x
    assert len(flips) >= 6
    assert all(math.copysign(1.0, a) != math.copysign(1.0, b) for a, b in zip(flips, flips[1:]))


def test_the_robots_turn_round_together():
    """Each robot's tactic keeps its own way; both read the same ball, so they flip on the same tick."""
    a, b = ex.PassingRallyTactic(), ex.PassingRallyTactic()
    for x in (-1.0, -0.4, 0.2, 0.8, 0.2, -0.4, -1.0):
        g = _game({"x": x - 0.1}, {"x": 0.0, "y": 0.5}, make_ball(x=x))
        a.sign, b.sign = ex.rally_direction(g, a.sign), ex.rally_direction(g, b.sign)
        assert a.sign == b.sign


def test_the_receiver_gets_open_upfield_in_the_lane_across_from_the_ball():
    g = _game({"x": -0.9, "y": 0.2}, {"x": -0.5, "y": 0.0}, make_ball(x=-0.8, y=0.2))
    assert ex.upfield_spot(g, Vector2D(-0.5, 0.0), 1.0) == Vector2D(-0.8 + ex.ADVANCE_M, -ex.LANE_Y_M)
    g = _game({"x": -0.9, "y": -0.2}, {"x": -0.5, "y": 0.0}, make_ball(x=-0.8, y=-0.2))
    assert ex.upfield_spot(g, Vector2D(-0.5, 0.0), 1.0) == Vector2D(-0.8 + ex.ADVANCE_M, ex.LANE_Y_M)


def test_with_the_ball_on_the_centre_line_the_receiver_keeps_its_own_lane():
    g = _game({"x": -0.9}, {"x": -0.5, "y": 0.4}, make_ball(x=-0.8, y=0.0))
    assert ex.upfield_spot(g, Vector2D(-0.5, 0.4), 1.0).y == pytest.approx(ex.LANE_Y_M)
    assert ex.upfield_spot(g, Vector2D(-0.5, -0.4), 1.0).y == pytest.approx(-ex.LANE_Y_M)


def test_heading_back_the_receiver_gets_open_towards_the_other_end():
    g = _game({"x": 0.9, "y": 0.2}, {"x": 0.0, "y": 0.0}, make_ball(x=1.0, y=0.2))
    spot = ex.upfield_spot(g, Vector2D(0.0, 0.0), -1.0)
    assert spot.x == pytest.approx(1.0 - ex.ADVANCE_M)
    assert spot.y == pytest.approx(-ex.LANE_Y_M)


def test_the_spot_stays_inside_the_field():
    g = _game({"x": -0.1}, {"x": 0.0}, make_ball(x=-0.0, y=1.0))
    half_width = g.field.half_width
    spot = ex.upfield_spot(g, Vector2D(0.0, 0.0), 1.0)
    assert abs(spot.y) <= half_width - ex._FIELD_MARGIN_M + 1e-9


@pytest.mark.parametrize("off_spot, is_set", [(ex._SET_M - 0.05, True), (ex._SET_M + 0.05, False)])
def test_the_passer_waits_until_the_partner_is_at_its_upfield_spot(off_spot, is_set):
    """A partner still and facing the ball but short of its spot got the ball straight back, and
    the rally went back and forth without going up the field (rsim, 2026-10-28)."""
    ball = make_ball(x=-0.8, y=0.2)
    spot = Vector2D(-0.8 + ex.ADVANCE_M, -ex.LANE_Y_M)
    at = Vector2D(spot.x - off_spot, spot.y)  # short of the spot, still a passing distance away
    facing = math.atan2(ball.p.y - at.y, ball.p.x - at.x)
    g = _game({"x": -0.9, "y": 0.2}, {"x": at.x, "y": at.y, "orientation": facing}, ball)
    partner = g.enemy_robots[0]
    assert pr.partner_ready(g, partner)
    assert ex.partner_set(g, partner, 1.0) is is_set
