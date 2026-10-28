"""`examples/passing_rally.py`: `pass_receive.py`'s pair passing up the field in two lanes and
shooting into the empty goal. Every rule here is a pure function of the world, as both sides read it."""

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


def _range_edge(game) -> float:
    """The x where shooting range starts: `SHOOT_RANGE_M` short of the attacked goal line."""
    return ex.goal_centre(game).x - ex.SHOOT_RANGE_M


def test_the_robot_on_the_ball_passes_out_of_range_and_shoots_in_range():
    edge = _range_edge(_game({"x": -0.1}, {"x": -1.0, "y": 0.5}, make_ball(x=0.0)))
    assert ex.rally_role(_game({"x": edge - 0.11}, {"x": -1.0, "y": 0.5}, make_ball(x=edge - 0.01)), 0) == "pass"
    assert ex.rally_role(_game({"x": edge - 0.09}, {"x": -1.0, "y": 0.5}, make_ball(x=edge + 0.01)), 0) == "shoot"


def test_in_range_the_target_is_the_goal_without_waiting_and_otherwise_the_partner():
    tactic = ex.PassingRallyTactic()
    g = _game({"x": 0.0}, {"x": -1.0, "y": 0.5}, make_ball(x=0.1))
    assert tactic.target(g, 0) == (g.enemy_robots[0].p, True)
    g = _game({"x": 1.0}, {"x": -1.0, "y": 0.5}, make_ball(x=1.1))
    assert tactic.target(g, 0) == (ex.goal_centre(g), False)


@pytest.mark.parametrize(
    "y_at_line, rolling, on_goal",
    [
        (0.0, 2.0, True),
        ("inside", 2.0, True),  # just inside the post, counting the ball's radius
        ("outside", 2.0, False),
        (0.0, -2.0, False),  # rolling away from the goal
        (0.0, ex._ROLLING_MPS * 0.9, False),  # too slow to be a shot
    ],
)
def test_a_shot_on_goal_is_a_ball_rolling_at_the_goal_mouth(y_at_line, rolling, on_goal):
    g = _game({"x": 0.0}, {"x": -1.0}, make_ball())
    half_mouth = g.field.half_goal_width + ex.BALL_RADIUS
    target_y = {"inside": half_mouth - 0.01, "outside": half_mouth + 0.01}.get(y_at_line, y_at_line)
    start = Vector2D(0.5, 0.0)
    line_x = ex.goal_centre(g).x
    heading = math.atan2(target_y - start.y, line_x - start.x)
    ball = make_ball(x=start.x, y=start.y, vx=rolling * math.cos(heading), vy=rolling * math.sin(heading))
    assert ex.shot_on_goal(_game({"x": 0.0}, {"x": -1.0}, ball)) is on_goal


def test_the_receiver_leaves_a_shot_alone():
    """A shot rolling past the partner into the mouth is not a pass to step onto."""
    ball = make_ball(x=0.6, y=0.0, vx=2.0)
    g = _game({"x": 1.0, "y": 0.05, "orientation": math.pi}, {"x": 0.3, "y": 0.0}, ball)
    assert ex.shot_on_goal(g)
    assert ex.rally_role(g, 0) != "receive"


def test_the_receiver_gets_open_upfield_in_the_lane_across_from_the_ball():
    g = _game({"x": -0.9, "y": 0.2}, {"x": -0.5, "y": 0.0}, make_ball(x=-0.8, y=0.2))
    assert ex.upfield_spot(g, Vector2D(-0.5, 0.0)) == Vector2D(-0.8 + ex.ADVANCE_M, -ex.LANE_Y_M)
    g = _game({"x": -0.9, "y": -0.2}, {"x": -0.5, "y": 0.0}, make_ball(x=-0.8, y=-0.2))
    assert ex.upfield_spot(g, Vector2D(-0.5, 0.0)) == Vector2D(-0.8 + ex.ADVANCE_M, ex.LANE_Y_M)


def test_with_the_ball_on_the_centre_line_the_receiver_keeps_its_own_lane():
    g = _game({"x": -0.9}, {"x": -0.5, "y": 0.4}, make_ball(x=-0.8, y=0.0))
    assert ex.upfield_spot(g, Vector2D(-0.5, 0.4)).y == pytest.approx(ex.LANE_Y_M)
    assert ex.upfield_spot(g, Vector2D(-0.5, -0.4)).y == pytest.approx(-ex.LANE_Y_M)


def test_in_shooting_range_the_partner_trails_behind_the_ball():
    g = _game({"x": 0.9, "y": 0.2}, {"x": 0.0, "y": 0.0}, make_ball(x=1.0, y=0.2))
    spot = ex.upfield_spot(g, Vector2D(0.0, 0.0))
    assert spot.x == pytest.approx(1.0 - ex.ADVANCE_M)
    assert spot.x < g.ball.p.x


def test_the_spot_stays_inside_the_field():
    g = _game({"x": -0.1}, {"x": 0.0}, make_ball(x=-0.0, y=1.0))
    half_width = g.field.half_width
    spot = ex.upfield_spot(g, Vector2D(0.0, 0.0))
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
    assert ex.partner_set(g, partner) is is_set
