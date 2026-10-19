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
from utama_core.tactics.press_and_contain import (
    PressAndContainMem,
    PressAndContainTactic,
)


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


def test_press_and_contain_applicable_hysteresis():
    tactic = PressAndContainTactic()

    # Enter threshold: 1.4 m, Exit threshold: 1.7 m
    # Ball at (0, 0).
    def _game_with_enemy_dist(dist: float) -> Game:
        friendly = {2: _robot(2, 1.0, 1.0, True), 3: _robot(3, 2.0, 2.0, True)}
        enemy = {7: _robot(7, dist, 0.0, False)}
        zero = Vector3D(0.0, 0.0, 0.0)
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=friendly,
            enemy_robots=enemy,
            ball=Ball(p=zero, v=zero, a=zero),
        )
        field = Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        )
        return Game(past=GameHistory(max_history=20), current=frame, field=field)

    # 1) Enemy at distance 1.55 m (between 1.4 m and 1.7 m). Initial is False -> False.
    assert not tactic.applicable(_game_with_enemy_dist(1.55))

    # 2) Enemy moves to 1.3 m (<= 1.4 m). Enters -> True.
    assert tactic.applicable(_game_with_enemy_dist(1.3))

    # 3) Enemy moves back to 1.55 m (> 1.4 m, <= 1.7 m). Remains -> True.
    assert tactic.applicable(_game_with_enemy_dist(1.55))

    # 4) Enemy moves to 1.75 m (> 1.7 m). Exits -> False.
    assert not tactic.applicable(_game_with_enemy_dist(1.75))

    # 5) Reset clears state.
    tactic.applicable(_game_with_enemy_dist(1.3))  # now True
    tactic.reset_applicable()
    assert not tactic.applicable(_game_with_enemy_dist(1.55))  # reset back to False


def test_presser_sticky_margin():
    tactic = PressAndContainTactic()
    ctx = TickContext(motion_controller=_NullMotionController())

    # Ball at (0, 0), enemy at (0.5, 0.0).
    # Friendly robot 2 at (1.5, 0.0) -> distance to enemy = 1.0 m.
    # Friendly robot 3 at (1.4, 0.0) -> distance to enemy = 0.9 m (0.1 m closer, < 0.2 m margin).
    def _game_with_positions(r2_x: float, r3_x: float) -> Game:
        friendly = {2: _robot(2, r2_x, 0.0, True), 3: _robot(3, r3_x, 0.0, True)}
        enemy = {7: _robot(7, 0.5, 0.0, False)}
        zero = Vector3D(0.0, 0.0, 0.0)
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=friendly,
            enemy_robots=enemy,
            ball=Ball(p=zero, v=zero, a=zero),
        )
        field = Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        )
        return Game(past=GameHistory(max_history=20), current=frame, field=field)

    mem = PressAndContainMem(presser_id=2)
    # Robot 3 is 0.1 m closer (< 0.2 m margin) -> robot 2 remains presser!
    _, new_mem = tactic.tick(_game_with_positions(1.5, 1.4), ctx, (2, 3), mem)
    assert new_mem.presser_id == 2

    # Robot 3 moves to (1.2, 0.0) -> distance 0.7 m (0.3 m closer, > 0.2 m margin) -> switches to 3!
    _, new_mem2 = tactic.tick(_game_with_positions(1.5, 1.2), ctx, (2, 3), mem)
    assert new_mem2.presser_id == 3
