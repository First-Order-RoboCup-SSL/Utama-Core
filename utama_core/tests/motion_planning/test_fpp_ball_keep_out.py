"""FastPathPlanner keeps a robot out of the ball's keep-out circle while the
other team takes a restart.

RR#6 low_block_vs_three_slot (2026-09-28): PreparePenaltyTheirsStep commanded
detour waypoints outside the circle, but the planner sidestepped teammates
queueing at the same waypoint on the ball side -- the ball is not an obstacle
to it -- and a defender reached 0.38 m from the ball, so KeepOutRule voided
the penalty.
"""

import math
from types import SimpleNamespace

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field
from utama_core.entities.game.game import Game
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import GameHistory
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner

_BALL = (2.25, 0.0)


def _game(command: RefereeCommand) -> Game:
    robot = Robot(
        id=2,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(3.1, 0.05),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )
    zero = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=1.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={2: robot},
        enemy_robots={},
        ball=Ball(p=Vector3D(*_BALL, 0), v=zero, a=zero),
        referee=SimpleNamespace(referee_command=command, designated_position=None),
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(10), current=frame, field=field)


def _waypoint_distance_to_ball(command: RefereeCommand) -> float:
    """Closest the straight run from the robot to the planner's waypoint comes to the ball."""
    start = (3.1, 0.05)
    planner = FastPathPlanner(env=None)
    waypoint = planner._path_to(_game(command), 2, (1.25, 0.05), STANDARD_FIELD_DIMS.full_field_bounds)
    sx, sy = waypoint[0] - start[0], waypoint[1] - start[1]
    t = min(1.0, max(0.0, ((_BALL[0] - start[0]) * sx + (_BALL[1] - start[1]) * sy) / (sx * sx + sy * sy)))
    return math.hypot(start[0] + sx * t - _BALL[0], start[1] + sy * t - _BALL[1])


@pytest.mark.parametrize(
    "command",
    [RefereeCommand.PREPARE_PENALTY_BLUE, RefereeCommand.DIRECT_FREE_BLUE, RefereeCommand.STOP],
)
def test_waypoint_stays_outside_the_circle_while_the_other_team_restarts(command):
    assert _waypoint_distance_to_ball(command) >= BALL_KEEP_OUT_DISTANCE - 1e-9


@pytest.mark.parametrize("command", [RefereeCommand.DIRECT_FREE_YELLOW, RefereeCommand.NORMAL_START])
def test_our_own_restart_and_open_play_still_route_near_the_ball(command):
    assert _waypoint_distance_to_ball(command) < BALL_KEEP_OUT_DISTANCE
