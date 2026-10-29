"""Our placement of a ball just outside the enemy box may enter the box and its keep-out.

Field test, 2026-10-27: the ball rested 1 cm outside the enemy box; the placer's approach
point, behind the ball, was inside it, so the planner clamped the placer to the keep-out line,
0.25 m short of the ball, for the whole placement.
"""

from types import SimpleNamespace

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field
from utama_core.entities.game.game import Game
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import GameHistory
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner
from utama_core.motion_planning.src.trajsampling.planner import (
    TrajectorySamplingPlanner,
)


def _game(command: RefereeCommand, ball_gap_m: float) -> Game:
    """Yellow (us) on the right, so the enemy box is on the left; the ball `ball_gap_m` outside it."""
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    box_max_x = max(c[0] for c in field.enemy_defense_area)
    zero = Vector2D(0, 0)
    robot = Robot(id=0, is_friendly=True, has_ball=False, p=Vector2D(0.0, 0.0), v=zero, a=zero, orientation=0.0)
    frame = GameFrame(
        ts=1.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: robot},
        enemy_robots={},
        ball=Ball(p=Vector3D(box_max_x + ball_gap_m, 0.0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        referee=SimpleNamespace(referee_command=command, designated_position=(0.0, 0.0)),
    )
    return Game(past=GameHistory(10), current=frame, field=field)


_PLANNERS = [FastPathPlanner(env=None), TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)]


@pytest.mark.parametrize("planner", _PLANNERS, ids=["fpp", "trajsampling"])
def test_our_placement_of_a_ball_just_outside_the_box_may_enter_it(planner):
    assert planner._enemy_defense_area_retrieval_exempt(_game(RefereeCommand.BALL_PLACEMENT_YELLOW, 0.01), 0)


@pytest.mark.parametrize("planner", _PLANNERS, ids=["fpp", "trajsampling"])
def test_exempt_up_to_the_keep_out_edge_and_not_past_it(planner):
    edge = OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
    assert planner._enemy_defense_area_retrieval_exempt(_game(RefereeCommand.BALL_PLACEMENT_YELLOW, edge - 1e-6), 0)
    assert not planner._enemy_defense_area_retrieval_exempt(_game(RefereeCommand.BALL_PLACEMENT_YELLOW, edge + 1e-6), 0)


@pytest.mark.parametrize("planner", _PLANNERS, ids=["fpp", "trajsampling"])
@pytest.mark.parametrize("command", [RefereeCommand.BALL_PLACEMENT_BLUE, RefereeCommand.STOP])
def test_their_placement_or_a_stop_keeps_us_out(planner, command):
    assert not planner._enemy_defense_area_retrieval_exempt(_game(command, 0.01), 0)
