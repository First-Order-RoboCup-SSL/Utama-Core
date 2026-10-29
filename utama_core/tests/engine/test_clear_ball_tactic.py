"""Unit tests for ClearBallTactic and in_danger hysteresis."""

from __future__ import annotations

import math

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.tactics.clear_ball import ClearBallTactic, in_danger


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


def _game(ball_x: float, ball_y: float, enemy_x: float, enemy_y: float) -> Game:
    # Field has half_length = 4.5.
    # my_team_is_right=True: our goal line is at x = +4.5.
    # ball at ball_x, distance from own goal line is 4.5 - ball_x.
    friendly = {
        0: _robot(0, 4.0, 0.0, True),
        1: _robot(1, 3.5, 0.0, True),
    }
    enemy = {
        5: _robot(5, enemy_x, enemy_y, False),
    }
    zero = Vector3D(0.0, 0.0, 0.0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=Ball(p=Vector3D(ball_x, ball_y, 0.0), v=zero, a=zero),
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_in_danger_pressure_hysteresis():
    tactic = ClearBallTactic()
    # Field half_length = 4.5. Own goal is x=4.5.
    # Ball at x=3.0 -> depth from own goal is 1.5 m (< 2.6 m danger depth).
    # 1) Enemy at distance 1.35 m (between 1.2 m enter and 1.5 m exit).
    # State was False -> should remain False.
    game = _game(ball_x=3.0, ball_y=0.0, enemy_x=3.0, enemy_y=1.35)
    assert not tactic.applicable(game)

    # 2) Enemy moves closer to 1.1 m (<= 1.2 m enter threshold).
    # Enters danger!
    game = _game(ball_x=3.0, ball_y=0.0, enemy_x=3.0, enemy_y=1.1)
    assert tactic.applicable(game)

    # 3) Enemy retreats back to 1.35 m (> 1.2 m but <= 1.5 m exit threshold).
    # Due to hysteresis, remains True!
    game = _game(ball_x=3.0, ball_y=0.0, enemy_x=3.0, enemy_y=1.35)
    assert tactic.applicable(game)

    # 4) Enemy retreats to 1.6 m (> 1.5 m exit threshold).
    # Exits danger!
    game = _game(ball_x=3.0, ball_y=0.0, enemy_x=3.0, enemy_y=1.6)
    assert not tactic.applicable(game)


def test_in_danger_depth_hysteresis():
    tactic = ClearBallTactic()
    # Enemy is always at distance 0.5 m from ball (<= 1.2 m).
    # Own goal is at x=4.5.
    # Enter depth: <= 2.6 m (ball_x >= 4.5 - 2.6 = 1.9 m)
    # Exit depth: > 3.0 m (ball_x < 4.5 - 3.0 = 1.5 m)

    # 1) Ball at x=1.7 m (depth = 2.8 m: between 2.6 m enter and 3.0 m exit).
    # Initial state False -> remains False.
    game = _game(ball_x=1.7, ball_y=0.0, enemy_x=1.7, enemy_y=0.5)
    assert not tactic.applicable(game)

    # 2) Ball moves deeper to x=2.0 m (depth = 2.5 m <= 2.6 m).
    # Enters danger!
    game = _game(ball_x=2.0, ball_y=0.0, enemy_x=2.0, enemy_y=0.5)
    assert tactic.applicable(game)

    # 3) Ball clears slightly to x=1.7 m (depth = 2.8 m <= 3.0 m).
    # Remains in danger!
    game = _game(ball_x=1.7, ball_y=0.0, enemy_x=1.7, enemy_y=0.5)
    assert tactic.applicable(game)

    # 4) Ball clears past exit depth to x=1.4 m (depth = 3.1 m > 3.0 m).
    # Exits danger!
    game = _game(ball_x=1.4, ball_y=0.0, enemy_x=1.4, enemy_y=0.5)
    assert not tactic.applicable(game)


def test_clear_ball_tactic_applicable_and_reset():
    tactic = ClearBallTactic()
    game = _game(ball_x=3.0, ball_y=0.0, enemy_x=3.0, enemy_y=1.1)
    assert tactic.applicable(game)

    # Ball moves to middle (1.35 m enemy distance): hysteresis keeps it applicable
    game2 = _game(ball_x=3.0, ball_y=0.0, enemy_x=3.0, enemy_y=1.35)
    assert tactic.applicable(game2)

    # Reset clears the state
    tactic.reset_applicable()
    assert not tactic.applicable(game2)
