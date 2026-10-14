import math
from unittest.mock import MagicMock

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.shared.pass_and_score_geometry import enemy_defense_area_hold_point
from utama_core.skills.src.go_to_ball import (
    _APPROACH_OVERSHOOT_M,
    _DRIBBLE_OVERSHOOT_M,
    _target_past_ball,
    go_to_ball,
)


def test_target_past_ball_overshoots_along_approach_line():
    ball = Vector2D(1.0, 0.0)

    target = _target_past_ball(ball, 0.0, 0.05)

    assert target.x == pytest.approx(1.05)
    assert target.y == pytest.approx(0.0)


def test_target_past_ball_overshoots_diagonal_approach():
    ball = Vector2D(1.0, 1.0)

    target = _target_past_ball(ball, math.atan2(4.0, 3.0), 0.05)

    assert target.x == pytest.approx(1.03)
    assert target.y == pytest.approx(1.04)


def test_dribbler_off_overshoot_is_smaller_than_dribbler_on():
    # _APPROACH_OVERSHOOT_M is currently tuned to 0 (no overshoot needed to
    # capture the ball without the dribbler on) — the real invariant this
    # test guards is the ordering, not one specific historical value.
    assert _DRIBBLE_OVERSHOOT_M > 0.0
    assert _APPROACH_OVERSHOOT_M < _DRIBBLE_OVERSHOOT_M


def _robot(rid: int, x: float, y: float, is_friendly: bool) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _game(friendly: dict, enemy: dict, ball_xy: tuple) -> Game:
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def _motion_controller() -> MagicMock:
    mc = MagicMock()
    mc.calculate.return_value = (Vector2D(0.0, 0.0), 0.0)
    return mc


def test_holds_outside_enemy_box_instead_of_chasing_ball_inside_it():
    """my_team_is_right=True -> enemy box is on the LEFT (negative x).

    Every `go_to_ball` caller (any tactic) must get this for free — see the
    module docstring for the oscillation this prevents.
    """
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    enemy_box_front_x = float(field.enemy_defense_area[1][0])
    ball_xy = (enemy_box_front_x - 0.2, 0.3)  # inside the enemy box

    friendly = {1: _robot(1, ball_xy[0] + 1.0, 0.0, True)}
    enemy = {0: _robot(0, ball_xy[0], ball_xy[1], False)}  # enemy keeper camped on the ball
    game = _game(friendly, enemy, ball_xy)
    mc = _motion_controller()

    go_to_ball(game=game, motion_controller=mc, robot_id=1)

    target = mc.calculate.call_args.kwargs["target_pos"]
    expected = enemy_defense_area_hold_point(game, ball_xy[1])
    assert target.x == pytest.approx(expected.x)
    assert target.y == pytest.approx(expected.y)
    assert target.x > ball_xy[0]  # the hold point is genuinely outside the box


def test_chases_ball_directly_when_ball_is_not_in_enemy_defense_area():
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {0: _robot(0, 3.0, 3.0, False)}
    game = _game(friendly, enemy, (1.0, 0.0))  # mid-field, not in either box
    mc = _motion_controller()

    go_to_ball(game=game, motion_controller=mc, robot_id=1)

    target = mc.calculate.call_args.kwargs["target_pos"]
    assert target.distance_to(Vector2D(1.0, 0.0)) < 0.5


@pytest.mark.parametrize("robot_xy", [(0.0, 0.0), (2.0, 1.0), (1.0, -1.5)])
def test_approaches_facing_the_ball(monkeypatch, robot_xy):
    """The kicker and dribbler are on the front: the robot faces the ball on its way in."""
    from utama_core.skills.src import go_to_ball as module

    captured = {}
    monkeypatch.setattr(module, "move", lambda **kwargs: captured.update(kwargs))
    game = _game({1: _robot(1, *robot_xy, True)}, {}, (1.0, 0.0))

    go_to_ball(game=game, motion_controller=_motion_controller(), robot_id=1, shield=False)

    assert captured["target_oren"] == pytest.approx(math.atan2(0.0 - robot_xy[1], 1.0 - robot_xy[0]))
