"""Regression: `PressAndContainTactic` won the ball and did nothing with it.

With an enemy adjacent the tactic stays applicable, so it kept the slot and its
presser kept running `block_attacker` against that enemy while holding the
ball, until the enemy took it back (65 of 309 PressAndContain losses in the
2026-09-24 round-robin had our robot in dribbler contact first). A robot in the
slot with the ball on its dribbler must clear it.
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
from utama_core.tactics.press_and_contain import PressAndContainTactic


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


def _game(winner_has_ball: bool) -> Game:
    # my_team_is_right: upfield is -x (orientation pi). Robot 2 has just won the
    # ball off enemy 7, who is still right next to it; robot 3 is farther away.
    friendly = {
        2: _robot(2, 0.0, 0.0, True, has_ball=winner_has_ball, orientation=math.pi),
        3: _robot(3, 1.0, 1.0, True),
    }
    enemy = {7: _robot(7, 0.25, 0.1, False), 8: _robot(8, -2.0, 1.0, False)}
    zero = Vector3D(0.0, 0.0, 0.0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=Ball(p=Vector3D(-0.09, 0.0, 0.0), v=zero, a=zero),
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_robot_that_won_the_ball_clears_it():
    tactic = PressAndContainTactic()
    game = _game(winner_has_ball=True)
    assert tactic.applicable(game)  # the enemy is still close: the slot stays pressing

    commands, mem = tactic.tick(
        game, TickContext(motion_controller=_NullMotionController()), (2, 3), tactic.initial_mem()
    )

    assert mem.presser_id == 2
    assert commands[2].kick == 1


def test_without_the_ball_the_presser_still_presses():
    tactic = PressAndContainTactic()
    commands, _ = tactic.tick(
        _game(winner_has_ball=False),
        TickContext(motion_controller=_NullMotionController()),
        (2, 3),
        tactic.initial_mem(),
    )
    assert commands[2].kick == 0
