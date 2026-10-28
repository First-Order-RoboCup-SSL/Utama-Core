"""`examples/real_pass_receive.py`: two robots, one per team, pass to each other. Each side reads
its role from the same world, so the two sides must always agree: one passer, one receiver."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from utama_core.shared.pass_and_score_geometry import reset_possession_state
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import real_pass_receive as ex
finally:
    sys.path.remove(str(EXAMPLES))


@pytest.fixture(autouse=True)
def _fresh_possession_state():
    reset_possession_state()
    yield
    reset_possession_state()


def _views(yellow: dict, blue: dict, ball=None, box_rules=False):
    """The same world as the yellow side and as the blue side sees it: (yellow game, blue game).
    The example's referee profiles have no defense-area rule, so the boxes are off by default."""
    ball = ball or make_ball()

    def _side(me: dict, other: dict, is_yellow: bool):
        return build_game(
            {0: make_robot(0, is_friendly=True, **me)},
            {0: make_robot(0, is_friendly=False, **other)},
            ball,
            my_team_is_yellow=is_yellow,
            my_team_is_right=not is_yellow,
            field_dims=ex.FIELD,
            defense_area_rules=box_rules,
        )

    return _side(yellow, blue, True), _side(blue, yellow, False)


def _roles(yellow: dict, blue: dict, ball=None) -> tuple[str, str]:
    g_yellow, g_blue = _views(yellow, blue, ball)
    return ex.pass_role(g_yellow, 0), ex.pass_role(g_blue, 0)


def test_the_robot_nearer_the_still_ball_passes_and_the_other_gets_open():
    assert _roles({"x": -0.3}, {"x": 0.8}) == ("pass", "get_open")
    assert _roles({"x": -0.8}, {"x": 0.3}) == ("get_open", "pass")


@pytest.mark.parametrize("gap", [0.0, ex._TIE_M - 0.01, ex._TIE_M + 0.01, -ex._TIE_M + 0.01, -ex._TIE_M - 0.01])
def test_exactly_one_passer_at_and_around_a_tie(gap):
    """Inside `_TIE_M` both robots are as near: yellow takes the ball, never both or neither."""
    roles = _roles({"x": -0.5}, {"x": 0.5 + gap})
    assert sorted(roles) == ["get_open", "pass"]
    if abs(gap) <= ex._TIE_M:
        assert roles == ("pass", "get_open")


def test_a_rolling_ball_is_received_by_the_robot_it_rolls_to_while_the_kicker_waits():
    rolling_to_blue = make_ball(x=0.0, vx=1.5)
    assert _roles({"x": -0.3}, {"x": 1.0, "y": 0.1}, rolling_to_blue) == ("wait", "receive")
    # Just under the rolling speed it is a still ball again: the nearer robot collects it.
    slow = make_ball(x=0.0, vx=ex._ROLLING_MPS - 0.01)
    assert _roles({"x": -0.3}, {"x": 1.0}, slow) == ("pass", "get_open")


def test_a_miss_rolling_between_them_goes_to_whichever_is_nearer_its_path():
    across = make_ball(x=0.0, y=0.0, vy=1.5)  # both robots are ahead of the ball, blue nearer its path
    assert _roles({"x": -0.6, "y": 1.0}, {"x": 0.2, "y": 0.8}, across) == ("wait", "receive")


def test_the_right_hand_receiver_steps_onto_a_pass_rolling_past_its_goal_side():
    """2026-10-27 field run, t = 12.9 s: the pass rolled 0.13 m past blue on its own goal's side.
    The shared 0.57 m box margin held blue at x <= 0.43, so it stood still and missed."""
    pass_ = make_ball(x=0.13, y=-0.37, vx=0.37, vy=0.91)
    _, g_blue = _views({"x": 0.0, "y": -0.6}, {"x": 0.39, "y": 0.63}, pass_)
    spot = ex.receive_spot(g_blue, 0)
    assert spot.x == pytest.approx(0.52, abs=0.01)


@pytest.mark.parametrize("past", [-0.01, 0.01])
def test_a_receive_spot_stops_box_margin_short_of_its_own_box(past):
    """Rolling straight into blue's own box, the spot stops `_BOX_MARGIN_M` short of its front
    while the box rules apply (the example's referee profiles leave them out)."""
    front_x = ex.FIELD.full_field_half_length - 2 * ex.FIELD.half_defense_area_depth
    edge_x = front_x - ex._BOX_MARGIN_M
    into_box = make_ball(x=0.0, vx=1.5)
    _, g_blue = _views({"x": -0.3}, {"x": edge_x + past, "y": 0.0}, into_box, box_rules=True)
    assert ex.receive_spot(g_blue, 0).x == pytest.approx(min(edge_x + past, edge_x))


def test_without_box_rules_a_receive_spot_may_be_in_its_own_box():
    into_box = make_ball(x=0.0, vx=1.5)
    _, g_blue = _views({"x": -0.3}, {"x": 1.2, "y": 0.0}, into_box)
    assert ex.receive_spot(g_blue, 0).x == pytest.approx(1.2)


def test_both_hold_while_the_ball_is_out_of_the_field():
    out = make_ball(x=0.0, y=ex.FIELD.full_field_half_width + 0.05)
    assert _roles({"x": -0.3}, {"x": 0.8}, out) == ("hold", "hold")


def test_get_open_stands_pass_distance_away_on_its_own_side():
    ball = make_ball(x=0.0, y=0.3)
    g_yellow, _ = _views({"x": 0.0, "y": -0.2}, {"x": 0.3, "y": 0.4}, ball)
    spot = ex.get_open_spot(g_yellow, 0)
    assert spot.distance_to(ball.p.to_2d()) == pytest.approx(ex.PASS_DISTANCE, abs=0.05)
    assert spot.x == pytest.approx(0.0)  # straight down the line it already stands on


def test_get_open_near_a_line_turns_away_from_it_rather_than_standing_close():
    near_line = make_ball(x=0.0, y=ex.FIELD.full_field_half_width - 0.2)
    g_yellow, _ = _views({"x": 0.0, "y": 1.1}, {"x": 0.5}, near_line)  # behind the ball, at the line
    spot = ex.get_open_spot(g_yellow, 0)
    assert spot.distance_to(near_line.p.to_2d()) >= ex._OPEN_MIN_M
    assert abs(spot.y) <= ex.FIELD.full_field_half_width - ex._FIELD_MARGIN_M + 0.03


def test_get_open_keeps_its_side_when_its_full_distance_spot_is_clipped():
    """2026-10-27 field run, t=21.4 s: the own-box clamp clipped the receiver's spot to 0.91 m,
    under the old 0.96 m cut-off, and it ran to the far side of the ball (y +0.70)."""
    ball = make_ball(x=0.16, y=-0.18)
    g_yellow, _ = _views({"x": -0.42, "y": -0.79}, {"x": 0.22, "y": -0.11}, ball)
    spot = ex.get_open_spot(g_yellow, 0)
    assert spot.y < ball.p.y  # still below the ball, where the receiver is
    assert spot.distance_to(ball.p.to_2d()) >= ex._OPEN_MIN_M


@pytest.mark.parametrize("start_deg", [90.0, -90.0, 30.0, 150.0])
def test_a_passer_beside_the_ball_orbits_it_facing_it_towards_the_line_up(monkeypatch, start_deg):
    """Ball beside the passer, the partner to the right (aim 0): behind the ball is at 180 deg.
    Each target lies on the orbit, a step nearer that angle, and the passer faces the ball."""
    captured = {}

    def fake_move(game, mc, robot_id, target, oren, dribbling=False):
        captured.update(target=target, oren=oren)
        return "move"

    monkeypatch.setattr(ex, "move", fake_move)
    a = math.radians(start_deg)
    passer = make_robot(0, is_friendly=True, x=0.12 * math.cos(a), y=0.12 * math.sin(a))
    game = build_game({0: passer}, {0: _partner()}, make_ball(), field_dims=ex.FIELD)
    ex.PassReceiveTactic().tick(game, MagicMock(match_log=None), (0,), None)

    target = captured["target"]
    at = math.atan2(target.y, target.x)
    left_before = abs((math.pi - a + math.pi) % (2 * math.pi) - math.pi)
    left_after = abs((math.pi - at + math.pi) % (2 * math.pi) - math.pi)
    assert left_after == pytest.approx(max(0.0, left_before - ex._ORBIT_STEP_RAD), abs=1e-9)
    assert ex._APPROACH_M - 1e-9 <= target.mag() <= ex._ORBIT_M + 1e-9
    assert captured["oren"] == pytest.approx(math.atan2(-target.y, -target.x))


def test_a_passer_further_than_the_orbit_range_walks_round_instead(monkeypatch):
    captured = {}
    monkeypatch.setattr(ex, "move", lambda game, mc, rid, target, oren, dribbling=False: captured.update(t=target))
    passer = make_robot(0, is_friendly=True, x=0.0, y=ex._ORBIT_RANGE_M + 0.01)
    game = build_game({0: passer}, {0: _partner()}, make_ball(), field_dims=ex.FIELD)
    ex.PassReceiveTactic().tick(game, MagicMock(match_log=None), (0,), None)
    ball = make_ball().p.to_2d()
    assert captured["t"] != ex.orbit_point(ball, passer.p, 0.0)  # a detour waypoint, not an orbit point


def _partner(**kw):
    defaults = {"x": 1.2, "orientation": math.pi}  # 1.2 m from the ball at the centre, facing it
    return make_robot(0, is_friendly=False, **{**defaults, **kw})


@pytest.mark.parametrize(
    "partner, ready",
    [
        (_partner(), True),
        (_partner(vx=ex.READY_SPEED + 0.05), False),  # still walking
        (_partner(x=ex._MIN_PASS_M - 0.05), False),  # too close
        (_partner(orientation=math.pi - ex.READY_FACING_RAD - 0.05), False),  # not facing the ball
        (_partner(orientation=math.pi - ex.READY_FACING_RAD + 0.05), True),
    ],
)
def test_the_partner_is_ready_only_still_at_pass_distance_and_facing_the_ball(partner, ready):
    game = build_game({0: make_robot(0, is_friendly=True, x=-0.1)}, {0: partner}, make_ball(), field_dims=ex.FIELD)
    assert ex.partner_ready(game, partner) is ready


@pytest.mark.parametrize("partner_moving, kicks", [(False, True), (True, False)])
def test_the_passer_kicks_only_once_the_partner_is_set(partner_moving, kicks):
    """Ball on the kicker (sensor), aimed at the partner: kick when it is set, else hold the ball."""
    passer = make_robot(0, is_friendly=True, x=-0.1, orientation=0.0, has_ball=True)
    partner = _partner(vy=0.5 if partner_moving else 0.0)
    game = build_game({0: passer}, {0: partner}, make_ball(), field_dims=ex.FIELD)
    ctx = MagicMock(match_log=None)
    commands, _ = ex.PassReceiveTactic().tick(game, ctx, (0,), None)
    assert bool(commands[0].kick) is kicks
    if not kicks:
        assert commands[0].dribble  # holds the ball meanwhile
    ctx.motion_controller.calculate.assert_not_called()
