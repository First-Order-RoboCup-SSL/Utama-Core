"""FastPathPlanner never takes a robot already inside the ball's keep-out circle closer to
the ball during a restart that keeps it out.

Moment placement_corner_robot_drives_onto_ball (tournament_20261017_002337): during the
opponent's ball placement, robot 1 sat 0.16 m from the ball in a corner, boxed in by enemies 4
and 5. The carrot the planner gave it swung round enemy 4 and ran within 0.13 m of the ball
(the robot's own run toward that carrot closes on the ball). Step 8 guarded only robots
outside the circle.

The state is one the moment reaches about half a second in, not its start: from rest the
carrot at the start brings the robot only 0.003 m closer, inside the tolerance.
"""

import numpy as np

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.global_utils.math_utils import distance_point_to_segment
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

_BALL = (-3.747805, -2.497416)
_US = (-3.909004, -2.467464)
_TARGET = (-4.4, -1.891450)  # the referee's push-out point
# All twelve robots of the captured state (id: x, y, orientation). The carrot depends on the
# far ones too: with only the robots within 1.5 m the planner gives a straight run that
# never nears the ball.
_FRIENDLY = {
    0: (4.40873, 0.05898, -3.092253),
    1: (-3.909004, -2.467464, 2.206669),
    2: (-1.797528, 0.747956, -2.172401),
    3: (-1.801786, -0.748043, -2.449482),
    4: (-3.358823, -2.494843, -3.6e-05),
    5: (-3.419104, -1.959768, 0.973707),
}
_ENEMY = {
    0: (-4.445358, -0.409642, -0.002635),
    1: (-2.546535, -0.71166, -2.220263),
    2: (-3.958383, -1.727368, 1.734304),
    3: (-2.36478, 0.621993, -2.121389),
    4: (-3.953127, -2.293594, -2.541111),
    5: (-4.216878, -2.250004, 1.885296),
}


def test_carrot_does_not_bring_a_robot_inside_the_ball_keep_out_closer_to_the_ball():
    game = build_game(
        {rid: make_robot(rid, is_friendly=True, x=x, y=y, orientation=o) for rid, (x, y, o) in _FRIENDLY.items()},
        {rid: make_robot(rid, is_friendly=False, x=x, y=y, orientation=o) for rid, (x, y, o) in _ENEMY.items()},
        make_ball(*_BALL),
        referee_command=RefereeCommand.BALL_PLACEMENT_BLUE,
    )
    carrot = FastPathPlanner(env=None)._path_to(game, 1, _TARGET, STANDARD_FIELD_DIMS.full_field_bounds)
    ball, us = np.array(_BALL), np.array(_US)
    start_gap = np.linalg.norm(us - ball)
    # Smoothing may graze the start gap by a centimetre; without the fix the carrot runs
    # about 0.02 m closer than that, and the robot reaches 0.11 m from the ball.
    assert distance_point_to_segment(ball, us, carrot) >= start_gap - 0.01
