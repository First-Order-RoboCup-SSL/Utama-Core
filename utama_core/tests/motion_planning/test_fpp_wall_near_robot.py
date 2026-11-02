"""FastPathPlanner drives straight at an in-field target from beside a touchline.

Lab field run 2026-10-28 (20261028_192923_real_pass_receive, blue 0): the passer stood
0.24 m from the touchline, inside the wall's clearance, lined up on a ball 0.30 m from it.
The wall counted as an obstacle between them (only a wall near the *target* was exempt),
so the planner sent the robot on a 0.7 m detour away from the ball, every time it tried
to drive onto it.
"""

import math

import pytest

from utama_core.config.field_params import FieldDimensions
from utama_core.entities.data.vector import Vector2D
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

# The lab field: 3 m x 2.25 m (`examples/shooting_drill.py`'s FIELD).
_LAB = FieldDimensions(
    full_field_half_length=1.5,
    full_field_half_width=1.125,
    half_defense_area_depth=0.25,
    half_defense_area_width=0.5,
    half_goal_width=0.4,
    center_circle_radius=0.3,
)
_HEADING = math.radians(32)


@pytest.mark.parametrize("ball_y", [-0.75, -0.80, -0.83, -0.90])
def test_a_robot_beside_the_touchline_drives_straight_onto_the_ball(ball_y):
    """The robot 0.155 m behind the ball along its heading: at ball_y -0.80 and -0.83 it is
    0.24 m and 0.21 m from the touchline (y = -1.125) and the ball 0.33 m and 0.30 m, the
    band where the wall was kept between them. Its next waypoint is the ball itself."""
    ball = Vector2D(-0.52, ball_y)
    me = ball - Vector2D(math.cos(_HEADING), math.sin(_HEADING)) * 0.155
    game = build_game(
        {0: make_robot(0, is_friendly=True, x=me.x, y=me.y, orientation=_HEADING)},
        {},
        make_ball(ball.x, ball.y),
        field_dims=_LAB,
        my_team_is_right=False,
        defense_area_rules=False,
    )

    carrot = FastPathPlanner(env=None)._path_to(game, 0, ball, game.field.field_bounds)

    assert Vector2D(float(carrot[0]), float(carrot[1])).distance_to(ball) < 1e-6
