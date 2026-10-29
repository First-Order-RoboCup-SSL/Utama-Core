"""FastPathPlanner lets a robot pressed against another robot (inside its clearance) drive
straight away from it.

Moment placement_in_corner_traps_defender (tournament_20261017_002337): a defender 0.18 m
from the opponent's placer, inside the 0.216 m clearance, could plan no segment out, the
straight one away included; the detour ran back through the placer and it never left.
"""

import numpy as np
import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.global_utils.math_utils import distance_point_to_segment
from utama_core.motion_planning.src.fastpathplanning.planner import (
    FastPathPlanner,
    _leaving,
)
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

_US, _PLACER, _TARGET = (4.28, 2.35), (4.1, 2.37), (4.36, 1.78)


def test_planner_drives_a_pressed_robot_away_from_what_it_is_pressed_against():
    game = build_game(
        {2: make_robot(2, is_friendly=True, x=_US[0], y=_US[1])},
        {3: make_robot(3, is_friendly=False, x=_PLACER[0], y=_PLACER[1])},
        make_ball(4.0, 2.5),
        referee_command=RefereeCommand.BALL_PLACEMENT_BLUE,
    )
    carrot = FastPathPlanner(env=None)._path_to(game, 2, _TARGET, STANDARD_FIELD_DIMS.full_field_bounds)
    start_gap = np.hypot(_US[0] - _PLACER[0], _US[1] - _PLACER[1])
    # The carrot is smoothed, not the straight run itself: allow it to graze 1 cm closer.
    # Without the fix it runs back past the placer, about 0.05 m from it.
    assert distance_point_to_segment(np.array(_PLACER), np.array(_US), carrot) >= start_gap - 0.01


_POINT = (np.array([0.0, 0.0]), np.array([0.0, 0.0]))  # a robot obstacle at the origin


@pytest.mark.parametrize(
    "start, target, leaving",
    [
        ((0.1, 0.0), (1.0, 0.0), True),  # inside, straight away
        ((0.1, 0.0), (0.1, 1.0), True),  # inside, square to it: never closer
        ((0.1, 0.0), (-1.0, 0.0), False),  # inside, through it
        ((0.1, 0.0), (0.0, 1.0), False),  # inside, closer on the way
        ((0.3, 0.0), (1.0, 0.0), False),  # outside the clearance: an ordinary obstacle
    ],
)
def test_leaving_is_inside_the_clearance_and_never_closer(start, target, leaving):
    assert _leaving(np.array(start), np.array(target), _POINT, clearance=0.216) is leaving
