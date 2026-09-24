"""Regression: two `ShadowAndMarkTactic` shadows pinned the ball between them.

`defend_parameter` holds both shadows on a line just outside our own box with
their dribblers on. A ball on that line got held by both and pushed around
until `ExcessiveDribblingRule` fired (14 fouls in the 2026-09-24 round-robin,
traced in decoy_and_overload_vs_high_press: robots 3 and 4 both `has_ball` for
4.5s). A shadow with the ball on its dribbler must clear it instead.
"""

from __future__ import annotations

import math

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


class _NullMotionController(MotionController):
    def __init__(self):
        super().__init__(mode="rsim")

    def calculate(self, game, robot_id, target_pos, target_oren):
        return Vector2D(0.0, 0.0), 0.0


def _robot(rid: int, x: float, y: float, is_friendly: bool, has_ball: bool = False, orientation: float = 0.0):
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=orientation,
    )


def test_shadow_holding_the_ball_clears_it_instead_of_defending_on_it():
    # my_team_is_right: our goal at +x, upfield is -x (orientation pi).
    friendly = {
        3: _robot(3, 3.3, 0.0, True, has_ball=True, orientation=math.pi),
        4: _robot(4, 3.3, 0.25, True),
    }
    enemy = {1: _robot(1, 2.8, 0.0, False)}  # close enough that the ball isn't "loose"
    zero = Vector3D(0.0, 0.0, 0.0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=Ball(p=Vector3D(3.21, 0.0, 0.0), v=zero, a=zero),
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    game = Game(past=GameHistory(max_history=20), current=frame, field=field)
    tactic = ShadowAndMarkTactic()

    commands, _ = tactic.tick(
        game, TickContext(motion_controller=_NullMotionController()), (3, 4), tactic.initial_mem()
    )

    assert commands[3].kick == 1
    assert commands[4].kick == 0
