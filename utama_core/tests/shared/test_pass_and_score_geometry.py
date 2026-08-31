"""Tests for the enemy-defense-area helpers in `pass_and_score_geometry`.

Mirrors of the existing own-defense-area helpers, added for
`LeadAndSupportTactic`'s enemy-box hold fix — see that tactic's test file
for the end-to-end behavior these enable.
"""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.shared.pass_and_score_geometry import (
    ball_in_enemy_defense_area,
    enemy_defense_area_hold_point,
)

_FIELD = Field(
    my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
)
# `enemy_defense_area`'s corners: [goal-line-x, ...], [front-x, ...], ... —
# the "front" edge (facing mid-field) is corner index 1's x, same as
# `in_enemy_defense_area`/`clamp_outside_enemy_defense_area` use internally.
_ENEMY_BOX_FRONT_X = float(_FIELD.enemy_defense_area[1][0])


def _robot(rid: int, x: float, y: float, is_friendly: bool) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _game(ball_xy: tuple) -> Game:
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, 0.0, 0.0, True)},
        enemy_robots={0: _robot(0, ball_xy[0], ball_xy[1], False)},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def test_ball_in_enemy_defense_area_true_when_inside_box():
    game = _game((_ENEMY_BOX_FRONT_X - 0.2, 0.0))
    assert bool(ball_in_enemy_defense_area(game)) is True


def test_ball_in_enemy_defense_area_false_when_outside_box():
    game = _game((0.0, 0.0))  # mid-field
    assert bool(ball_in_enemy_defense_area(game)) is False


def test_ball_in_enemy_defense_area_false_just_outside_front_edge():
    game = _game((_ENEMY_BOX_FRONT_X + 0.05, 0.0))
    assert bool(ball_in_enemy_defense_area(game)) is False


def test_enemy_defense_area_hold_point_stays_outside_box():
    game = _game((_ENEMY_BOX_FRONT_X - 0.2, 0.3))
    hold = enemy_defense_area_hold_point(game, at_y=0.3)
    # my_team_is_right=True -> enemy box is on the left -> outside means x > front edge.
    assert hold.x > _ENEMY_BOX_FRONT_X
    assert hold.y == 0.3


def test_enemy_defense_area_hold_point_clamps_y_inside_box_width():
    game = _game((_ENEMY_BOX_FRONT_X - 0.2, 0.0))
    half_width = STANDARD_FIELD_DIMS.half_defense_area_width
    hold = enemy_defense_area_hold_point(game, at_y=half_width + 5.0)
    assert hold.y < half_width + 5.0
