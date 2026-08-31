"""`LeadAndSupportTactic`-specific behavior not covered by `test_all_tactics.py`'s table.

The enemy-box hold behavior itself (when the ball rests inside the enemy's
box, hold outside it instead of chasing an unreachable target) lives in
`go_to_ball` now, not this tactic — see `test_go_to_ball.py` for the
unit-level coverage of that. What's tested here is that
`LeadAndSupportTactic`'s leader, having no ball, actually delegates to
`go_to_ball` (rather than some tactic-specific approach logic that would
bypass the fix) — an integration-level check, not a re-test of the
geometry.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.shared.pass_and_score_geometry import enemy_defense_area_hold_point
from utama_core.tactics.lead_and_support import LeadAndSupportTactic


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


def _game(friendly: dict, enemy: dict, ball_xy: tuple) -> Game:
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def _ctx() -> TickContext:
    mc = MagicMock()
    mc.calculate.return_value = (Vector2D(0.0, 0.0), 0.0)
    return TickContext(motion_controller=mc, match_log=None)


def test_leader_holds_outside_box_when_ball_is_in_enemy_defense_area():
    """my_team_is_right=True -> enemy goal/box is on the LEFT (negative x)."""
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    enemy_box_front_x = float(field.enemy_defense_area[1][0])
    ball_xy = (enemy_box_front_x - 0.2, 0.0)  # inside the enemy box

    friendly = {1: _robot(1, ball_xy[0] + 1.0, 0.0, True)}  # not holding the ball, some distance away
    enemy = {0: _robot(0, ball_xy[0], ball_xy[1], False)}  # enemy keeper camped on the ball

    game = _game(friendly, enemy, ball_xy)
    ctx = _ctx()
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()

    commands, mem = tactic.tick(game, ctx, robot_ids=(1,), mem=mem)

    assert 1 in commands
    target = ctx.motion_controller.calculate.call_args.kwargs["target_pos"]
    expected = enemy_defense_area_hold_point(game, ball_xy[1])
    assert target.x == pytest.approx(expected.x)
    assert target.y == pytest.approx(expected.y)
    # The hold point itself must actually be outside the box, not just a
    # different point that happens to still be inside it.
    assert target.x > ball_xy[0]


def test_leader_chases_ball_directly_when_ball_is_not_in_enemy_defense_area():
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {0: _robot(0, 3.0, 3.0, False)}
    game = _game(friendly, enemy, (1.0, 0.0))  # mid-field, not in either box

    ctx = _ctx()
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, ctx, robot_ids=(1,), mem=mem)

    assert 1 in commands
    target = ctx.motion_controller.calculate.call_args.kwargs["target_pos"]
    # go_to_ball targets at (or an overshoot past) the ball itself, not a
    # defense-area hold point far from it.
    assert target.distance_to(Vector2D(1.0, 0.0)) < 0.5
