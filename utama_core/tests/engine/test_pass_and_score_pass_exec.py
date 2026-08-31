"""Tests for `_pass_and_score._pass_exec`'s `lane_blocked` reporting.

Found live: `GiveAndGoTactic` (via the shared `_pass_exec`) kept aiming a
hop at a receiver for 4+ seconds after two enemies settled onto the direct
carrier-receiver line, producing a highly interceptable pass every tick
with no way for the caller to notice and abandon early — see
`build_tiki_taka_kernel_strategy` vs `build_counter_press_kernel_strategy`,
t≈23s, `give_and_go.pass_target` locked on receiver 3 while enemies 1/5
sat ~0.15m off the direct line (well inside `segment_blocked`'s own
default clearance).
"""

from __future__ import annotations

import dataclasses

import pytest

from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.tactics._pass_and_score import _pass_exec


@pytest.fixture
def runner():
    # Same real-`StrategyRunner`/rsim fixture as `test_all_tactics.py`'s own
    # `runner_factory` (not shared via conftest — duplicated here rather
    # than introducing new shared test infra for one extra file).
    from utama_core.run.strategy_runner import StrategyRunner
    from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy

    r = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=2,
        exp_ball=True,
    )
    yield r
    r.close()


def _ctx(runner) -> TickContext:
    motion_controller = runner.my.motion_controller(runner.mode, runner.rsim_env)
    return TickContext(motion_controller=motion_controller)


def _with_frame(game, friendly, enemy, ball_xy):
    frame = game.current
    ball = Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0.0), v=Vector3D(0.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0))
    new_frame = GameFrame(
        ts=frame.ts,
        my_team_is_yellow=frame.my_team_is_yellow,
        my_team_is_right=frame.my_team_is_right,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=ball,
        referee=frame.referee,
    )
    game.add_game_frame(new_frame)


def test_lane_blocked_true_when_enemies_sit_on_the_direct_line(runner):
    game = runner.my.game
    friendly = dict(game.current.friendly_robots)
    enemy = dict(game.current.enemy_robots)

    # Carrier holding the ball, receiver ~1m along the carrier's facing —
    # mirrors the t=23s replay geometry (carrier facing the receiver,
    # receiver reachable, but flanked close-in on the direct line).
    friendly[1] = dataclasses.replace(
        friendly[1], has_ball=True, p=Vector2D(0.709, -0.535), orientation=Vector2D(0, 0).angle_to(Vector2D(1, 0.1))
    )
    friendly[2] = dataclasses.replace(friendly[2], has_ball=False, p=Vector2D(0.16, -0.41))
    enemy_ids = list(enemy.keys())[:2]
    enemy[enemy_ids[0]] = dataclasses.replace(enemy[enemy_ids[0]], p=Vector2D(0.56, -0.65))
    enemy[enemy_ids[1]] = dataclasses.replace(enemy[enemy_ids[1]], p=Vector2D(0.37, -0.30))

    _with_frame(game, friendly, enemy, ball_xy=(0.6, -0.495))
    game = runner.my.game

    _commands, _pass_complete, lane_blocked = _pass_exec(game, _ctx(runner), passer_id=1, receiver_id=2)

    assert lane_blocked is True


def test_lane_blocked_false_with_a_clear_lane(runner):
    game = runner.my.game
    friendly = dict(game.current.friendly_robots)
    enemy = dict(game.current.enemy_robots)

    friendly[1] = dataclasses.replace(
        friendly[1], has_ball=True, p=Vector2D(0.0, 0.0), orientation=Vector2D(0, 0).angle_to(Vector2D(1, 0))
    )
    friendly[2] = dataclasses.replace(friendly[2], has_ball=False, p=Vector2D(2.0, 0.0))
    # Enemies far off to the side, nowhere near the carrier-receiver line.
    enemy_ids = list(enemy.keys())[:2]
    enemy[enemy_ids[0]] = dataclasses.replace(enemy[enemy_ids[0]], p=Vector2D(1.0, 3.0))
    enemy[enemy_ids[1]] = dataclasses.replace(enemy[enemy_ids[1]], p=Vector2D(1.0, -3.0))

    _with_frame(game, friendly, enemy, ball_xy=(0.09, 0.0))
    game = runner.my.game

    _commands, _pass_complete, lane_blocked = _pass_exec(game, _ctx(runner), passer_id=1, receiver_id=2)

    assert lane_blocked is False
