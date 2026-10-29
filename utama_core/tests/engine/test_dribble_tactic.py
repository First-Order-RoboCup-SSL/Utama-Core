"""DribbleTactic's release: it lets go of the ball at the end of each segment."""

from __future__ import annotations

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.tactics.dribble import DribbleMem, DribbleTactic


def _game(ball_distance: float) -> Game:
    zero2, zero3 = Vector2D(0, 0), Vector3D(0.0, 0.0, 0.0)
    robot = Robot(id=0, is_friendly=True, has_ball=False, p=Vector2D(0, 0), v=zero2, a=zero2, orientation=0.0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: robot},
        enemy_robots={},
        ball=Ball(p=Vector3D(ball_distance, 0.0, 0.0), v=zero3, a=zero3),
    )
    dims = GREAT_EXHIBITION_FIELD_DIMS
    field = Field(my_team_is_right=True, field_dims=dims, field_bounds=dims.full_field_bounds)
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def _releasing_mem() -> DribbleMem:
    return DribbleMem(
        final_target=Vector2D(1.0, 0.6), dribble_limit=0.5, segment_target=Vector2D(1.0, 0.6), releasing=True
    )


def test_letting_go_backs_away_from_a_ball_that_stopped_close():
    # rsim, 2026-10-22: the ball stopped 11 cm away, short of the 13 cm the release waits for;
    # the robot stood still waiting, for good. It backs away, dribbler off, until the ball is clear.
    commands, mem = DribbleTactic().tick(_game(0.11), TickContext(motion_controller=None), (0,), _releasing_mem())
    assert commands[0].local_forward_vel < 0 and not commands[0].dribble
    assert mem.releasing

    commands, mem = DribbleTactic().tick(_game(0.14), TickContext(motion_controller=None), (0,), _releasing_mem())
    assert not mem.releasing and mem.dribbled_distance == -1.0  # clear: it goes back for the ball
