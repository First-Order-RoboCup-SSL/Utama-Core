"""`examples/save_and_counter.py`: each robot keeps its goal or attacks the other's. Both read their
role from the same world, so the two must always agree: one attacker, one keeper."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D
from utama_core.motion_planning.src.common.pose_ahead import PoseAhead
from utama_core.shared.pass_and_score_geometry import reset_possession_state
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import save_and_counter as ex
finally:
    sys.path.remove(str(EXAMPLES))


@pytest.fixture(autouse=True)
def _fresh_possession_state():
    reset_possession_state()
    yield
    reset_possession_state()


def _views(yellow: dict, blue: dict, ball=None):
    """The same world as yellow (defends -x) and as blue (defends +x) sees it."""
    ball = ball or make_ball()

    def _side(me: dict, other: dict, is_yellow: bool):
        return build_game(
            {0: make_robot(0, is_friendly=True, **me)},
            {0: make_robot(0, is_friendly=False, **other)},
            ball,
            my_team_is_yellow=is_yellow,
            my_team_is_right=not is_yellow,
            field_dims=ex.FIELD,
            defense_area_rules=False,
        )

    return _side(yellow, blue, True), _side(blue, yellow, False)


def _roles(yellow: dict, blue: dict, ball=None) -> tuple[str, str]:
    g_yellow, g_blue = _views(yellow, blue, ball)
    return ex.play_role(g_yellow, 0), ex.play_role(g_blue, 0)


def test_the_robot_nearer_the_still_ball_attacks_and_the_other_keeps():
    assert _roles({"x": -0.3}, {"x": 1.2}) == ("attack", "keep")
    assert _roles({"x": -1.2}, {"x": 0.3}) == ("keep", "attack")


@pytest.mark.parametrize("gap", [0.0, ex._TIE_M - 0.01, -ex._TIE_M + 0.01])
def test_a_tie_for_the_still_ball_goes_to_yellow_so_exactly_one_attacks(gap):
    assert _roles({"x": -0.5}, {"x": 0.5 + gap}) == ("attack", "keep")


def test_a_keeper_that_saves_the_ball_attacks_and_the_shooter_goes_back_to_keep():
    """The point of the example: blue's keeper holds the ball at its goal (+x), yellow (who shot)
    is nearer the middle; blue now attacks, yellow keeps."""
    ball = make_ball(x=1.31, y=0.0)
    assert _roles({"x": 0.6}, {"x": 1.4, "orientation": math.pi, "has_ball": True}, ball) == ("keep", "attack")


@pytest.mark.parametrize("vx", [1.5, -1.5])
def test_a_ball_rolling_towards_a_goal_keeps_that_goal_s_robot_in_it(vx):
    """A shot at blue's goal (+x) passing near blue's keeper must not pull it out to chase the
    ball: it keeps, though nearer the ball, and the shooter follows up. Mirrored for yellow."""
    ball = make_ball(x=0.6 if vx > 0 else -0.6, y=0.0, vx=vx)
    yellow_x, blue_x = (-0.2, 1.0) if vx > 0 else (-1.0, 0.2)
    roles = _roles({"x": yellow_x}, {"x": blue_x}, ball)
    assert roles == (("attack", "keep") if vx > 0 else ("keep", "attack"))


def test_an_opponent_on_the_ball_sends_the_other_robot_back_to_keep():
    ball = make_ball(x=0.0, y=0.0)
    """Yellow is nearer the middle, but blue is on the ball: yellow keeps."""
    assert _roles({"x": 0.3, "orientation": math.pi}, {"x": -0.1}, ball)[0] == "keep"


def test_a_ball_out_of_the_field_holds_both():
    ball = make_ball(x=ex.FIELD.full_field_half_length + 0.1, y=0.0)
    assert _roles({"x": -0.5}, {"x": 0.5}, ball) == ("hold", "hold")


def _carry(x: float):
    """Yellow (attacking +x) holding the ball at (x, 0.3), blue in its goal."""
    me = make_robot(0, is_friendly=True, x=x, y=0.3, has_ball=True)
    game = build_game(
        {0: me},
        {0: make_robot(0, is_friendly=False, x=1.4, y=0.0)},
        make_ball(x=x + 0.09, y=0.3),
        my_team_is_yellow=True,
        my_team_is_right=False,
        field_dims=ex.FIELD,
        defense_area_rules=False,
    )
    mc = MagicMock()
    mc.calculate.return_value = (Vector2D(1.0, 0.0), 0.0)  # full speed: the carry caps it
    mc.pose_ahead = PoseAhead(0.0)
    mc.carry_speed = mc.carry_turn = None  # as in the sims: the example's own carry cap is under test
    targets = []
    real_move = ex.move

    def spy(game, motion_controller, robot_id, target, *a, **kw):
        targets.append(target)
        return real_move(game, motion_controller, robot_id, target, *a, **kw)

    ex.move = spy
    try:
        command = ex.KeepOrAttackTactic()._carry(game, TickContext(motion_controller=mc), 0)
    finally:
        ex.move = real_move
    return command, targets


def test_the_carry_ends_inside_the_shooting_range_not_on_its_edge():
    """rsim: aimed at the range itself, the attacker stopped 1 mm outside it and stood there with
    the ball, the shooter never taking over. Measured along the field, as the shooter's sideways
    moves for a shot must not take it back out of range."""
    goal_x = ex.FIELD.full_field_half_length
    command, targets = _carry(goal_x - ex.SHOOT_RANGE_M - 0.001)
    assert command is not None
    (target,) = targets
    assert goal_x - target.x == pytest.approx(ex.SHOOT_RANGE_M - ex._CARRY_INTO_RANGE_M)
    assert target.y == pytest.approx(0.3)  # straight up the field, not towards the goal centre
    assert math.hypot(command.local_forward_vel, command.local_left_vel) <= ex.CARRY_SPEED_MPS + 1e-9


def test_in_range_the_shooter_takes_over():
    command, _ = _carry(ex.FIELD.full_field_half_length - ex.SHOOT_RANGE_M + 0.001)
    assert command is None


def test_no_carry_without_the_ball():
    tactic = ex.KeepOrAttackTactic()
    game, _ = _views({"x": -1.0}, {"x": 1.0})
    assert tactic._carry(game, MagicMock(), 0) is None
