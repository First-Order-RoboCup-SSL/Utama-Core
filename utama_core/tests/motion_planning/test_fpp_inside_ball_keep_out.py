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
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
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


def test_a_robot_just_inside_the_ball_keep_out_is_sent_out_of_it_not_onto_its_edge():
    """counter_flow_vs_tiki_taka_plus (tournament_20261017_151052), 252-267 s: in the other
    team's kick-off a robot 0.795 m from the ball, just inside the 0.8 m circle, was sent by
    the kick-off step round the circle toward its spot in its own half. That run is a chord
    that closes on the ball, so the planner replaced it with the push-out point on the circle,
    0.005 m away: the robot parked there, still inside, and the kick-off waited the referee's
    full 15 s cap. The push-out must take it clear of the circle."""
    us = (0.762, -0.225)
    round_the_circle = (0.8 * np.cos(np.radians(-76)), 0.8 * np.sin(np.radians(-76)))
    game = build_game(
        {4: make_robot(4, is_friendly=True, x=us[0], y=us[1])},
        {0: make_robot(0, is_friendly=False, x=0.12, y=0.0)},
        make_ball(0.0, 0.0),
        my_team_is_right=False,
        my_team_is_yellow=False,
        referee_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
    )
    carrot = np.asarray(
        FastPathPlanner(env=None)._path_to(game, 4, round_the_circle, STANDARD_FIELD_DIMS.full_field_bounds)
    )
    assert np.linalg.norm(carrot) > BALL_KEEP_OUT_DISTANCE + 0.02
    assert distance_point_to_segment(np.zeros(2), np.array(us), carrot) >= np.linalg.norm(us) - 1e-6
