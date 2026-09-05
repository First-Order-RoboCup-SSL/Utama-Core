import math
from unittest.mock import MagicMock

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.skills.src.utils.move_utils import turn_on_spot

PIVOT_RADIUS = ROBOT_RADIUS + BALL_RADIUS


def _robot(rid: int, x: float, y: float, is_friendly: bool, orientation: float = 0.0, has_ball: bool = False) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=orientation,
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


def _motion_controller(angular_vel: float) -> MagicMock:
    mc = MagicMock()
    mc.calculate.return_value = (Vector2D(0.0, 0.0), angular_vel)
    return mc


def test_turn_on_spot_pivots_around_ball_when_clear_of_obstacles():
    # Robot facing +x, holding the ball, turning left (positive angular_vel).
    # Local-left pivot compensation should be the standard -angular_vel * PIVOT_RADIUS,
    # with no nearby enemy to suppress it.
    friendly = {1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=True)}
    enemy = {0: _robot(0, 5.0, 5.0, False)}  # far away
    game = _game(friendly, enemy, (ROBOT_RADIUS + BALL_RADIUS, 0.0))
    mc = _motion_controller(angular_vel=1.0)

    cmd = turn_on_spot(game=game, motion_controller=mc, robot_id=1, target_oren=math.pi / 2)

    assert cmd.local_left_vel == pytest.approx(-1.0 * PIVOT_RADIUS)


def test_turn_on_spot_suppresses_pivot_push_into_a_wedged_enemy():
    """Reproduces the traced COMMITTED_FROZEN deadlock from
    split_shape_vs_switch_of_play_RK: a robot pivoting on the ball while
    body-to-body with an enemy kept commanding a lateral push straight into
    the enemy every tick. Box2D's contact resolution cancelled the motion,
    the geometry never changed, and the same command (and freeze) recurred
    forever.

    Robot faces +x at the origin, holding the ball. For angular_vel=+1.0 the
    pivot compensation (-angular_vel * PIVOT_RADIUS, local-left) is negative,
    i.e. global -y for this orientation. Placing the enemy at (0, -small)
    puts it directly in that push direction and within contact distance.
    """
    friendly = {1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=True)}
    enemy = {0: _robot(0, 0.0, -0.15, False)}  # within 2*ROBOT_RADIUS, directly along the pivot push
    game = _game(friendly, enemy, (ROBOT_RADIUS + BALL_RADIUS, 0.0))
    mc = _motion_controller(angular_vel=1.0)  # pivot push is -y (into the enemy)

    cmd = turn_on_spot(game=game, motion_controller=mc, robot_id=1, target_oren=math.pi / 2)

    assert cmd.local_left_vel == pytest.approx(0.0)
    assert cmd.angular_vel == pytest.approx(1.0)  # rotation itself is preserved


def test_turn_on_spot_keeps_pivot_push_away_from_a_wedged_enemy():
    """Same wedged geometry, but turning the other way -- the pivot push
    points away from the enemy (global +y), so it should NOT be suppressed.
    """
    friendly = {1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=True)}
    enemy = {0: _robot(0, 0.0, -0.15, False)}
    game = _game(friendly, enemy, (ROBOT_RADIUS + BALL_RADIUS, 0.0))
    mc = _motion_controller(angular_vel=-1.0)  # pivot push is +y (away from the enemy)

    cmd = turn_on_spot(game=game, motion_controller=mc, robot_id=1, target_oren=-math.pi / 2)

    assert cmd.local_left_vel == pytest.approx(1.0 * PIVOT_RADIUS)
