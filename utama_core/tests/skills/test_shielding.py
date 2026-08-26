"""Tests for `utama_core.skills.src.shielding`, extracted out of `go_to_ball.py`."""

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.skills.src.shielding import (
    COMMIT_RANGE,
    CONTEST_RANGE,
    nearest_contesting_enemy,
    shielded_approach_angle,
)


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


def test_no_contesting_enemy_returns_none():
    game = _game({0: _robot(0, 0.0, 0.0, True)}, {}, (1.0, 0.0))
    assert nearest_contesting_enemy(game, Vector2D(1.0, 0.0)) is None


def test_enemy_outside_contest_range_is_ignored():
    game = _game(
        {0: _robot(0, 0.0, 0.0, True)},
        {0: _robot(0, 1.0 + CONTEST_RANGE + 1.0, 0.0, False)},
        (1.0, 0.0),
    )
    assert nearest_contesting_enemy(game, Vector2D(1.0, 0.0)) is None


def test_enemy_inside_contest_range_is_returned():
    enemy_pos = Vector2D(1.0 + CONTEST_RANGE / 2, 0.0)
    game = _game({0: _robot(0, 0.0, 0.0, True)}, {0: _robot(0, enemy_pos.x, enemy_pos.y, False)}, (1.0, 0.0))
    result = nearest_contesting_enemy(game, Vector2D(1.0, 0.0))
    assert result == enemy_pos


def test_nearest_contesting_enemy_picks_closest_of_several():
    ball = Vector2D(0.0, 0.0)
    far = Vector2D(0.4, 0.0)
    near = Vector2D(0.1, 0.0)
    game = _game(
        {0: _robot(0, 5.0, 5.0, True)},
        {0: _robot(0, far.x, far.y, False), 1: _robot(1, near.x, near.y, False)},
        (0.0, 0.0),
    )
    assert nearest_contesting_enemy(game, ball) == near


def test_shielded_approach_angle_shields_when_enemy_contesting_and_far_from_ball():
    ball = Vector2D(0.0, 0.0)
    robot = Vector2D(-5.0, 0.0)  # far from ball, beyond COMMIT_RANGE
    enemy = Vector2D(0.3, 0.0)  # within CONTEST_RANGE
    game = _game({0: _robot(0, robot.x, robot.y, True)}, {0: _robot(0, enemy.x, enemy.y, False)}, (0.0, 0.0))

    angle, shielding = shielded_approach_angle(game, robot, ball)

    assert shielding is True
    assert angle == pytest.approx(enemy.angle_to(ball))


def test_shielded_approach_angle_commits_direct_once_close_to_ball():
    ball = Vector2D(0.0, 0.0)
    robot = Vector2D(COMMIT_RANGE / 2, 0.0)  # inside COMMIT_RANGE
    enemy = Vector2D(0.3, 0.3)  # still within CONTEST_RANGE
    game = _game({0: _robot(0, robot.x, robot.y, True)}, {0: _robot(0, enemy.x, enemy.y, False)}, (0.0, 0.0))

    angle, shielding = shielded_approach_angle(game, robot, ball)

    assert shielding is False
    assert angle == pytest.approx(robot.angle_to(ball))


def test_shielded_approach_angle_direct_when_no_contesting_enemy():
    ball = Vector2D(0.0, 0.0)
    robot = Vector2D(-5.0, 0.0)
    game = _game({0: _robot(0, robot.x, robot.y, True)}, {}, (0.0, 0.0))

    angle, shielding = shielded_approach_angle(game, robot, ball)

    assert shielding is False
    assert angle == pytest.approx(robot.angle_to(ball))
