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
import math

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


def _with_frame(game, friendly, enemy, ball_xy, ball_v=(0.0, 0.0)):
    frame = game.current
    ball = Ball(
        p=Vector3D(ball_xy[0], ball_xy[1], 0.0), v=Vector3D(ball_v[0], ball_v[1], 0.0), a=Vector3D(0.0, 0.0, 0.0)
    )
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


def test_passer_aims_at_a_receiver_in_place_not_at_the_receive_point(runner):
    """A receiver within `at_target`'s 0.08 m of the receive point stops moving, so a
    pass aimed at the receive point reaches it off-centre: here 0.06 m to the side at
    0.55 m, about 6 degrees, which strikes the side of the dribbler and deflects
    (receptions from 10 to 20 degrees off caught 7% in tournament_20260924_092119).
    The passer faces the receive point exactly and must still turn onto the receiver."""
    game = runner.my.game
    friendly = dict(game.current.friendly_robots)
    enemy = dict(game.current.enemy_robots)

    friendly[1] = dataclasses.replace(friendly[1], has_ball=True, p=Vector2D(0.0, 0.0), orientation=0.0)
    # The receive point is (0.6, 0) (0.5 m minimum along the passer's heading, snapped);
    # the receiver is 0.078 m from it, so it counts as in place.
    friendly[2] = dataclasses.replace(friendly[2], has_ball=False, p=Vector2D(0.55, 0.06), orientation=math.pi)
    enemy_ids = list(enemy.keys())[:2]
    enemy[enemy_ids[0]] = dataclasses.replace(enemy[enemy_ids[0]], p=Vector2D(1.0, 3.0))
    enemy[enemy_ids[1]] = dataclasses.replace(enemy[enemy_ids[1]], p=Vector2D(1.0, -3.0))

    _with_frame(game, friendly, enemy, ball_xy=(0.09, 0.0))
    game = runner.my.game

    commands, _pass_complete, _lane_blocked = _pass_exec(game, _ctx(runner), passer_id=1, receiver_id=2)

    assert not commands[1].kick
    assert commands[1].angular_vel > 0.0  # turning left, toward the receiver at +y


def _short_pass_with_ball_off_centre(runner, passer_orientation):
    """Passer at the origin, ball 0.036 m to its left on the dribbler, receiver in place
    0.71 m straight ahead and facing the ball: counter_flow_vs_low_block t=26.3
    (tournament_20260928_221404), where the ball sat 0.036 m off-centre and the pass
    passed 0.07 m beside the receiver's centre."""
    game = runner.my.game
    friendly = dict(game.current.friendly_robots)
    enemy = dict(game.current.enemy_robots)
    ball_xy = (0.09, 0.036)
    receiver_p = Vector2D(0.71, 0.0)
    friendly[1] = dataclasses.replace(friendly[1], has_ball=True, p=Vector2D(0.0, 0.0), orientation=passer_orientation)
    friendly[2] = dataclasses.replace(
        friendly[2], has_ball=False, p=receiver_p, orientation=receiver_p.angle_to(Vector2D(*ball_xy))
    )
    enemy_ids = list(enemy.keys())[:2]
    enemy[enemy_ids[0]] = dataclasses.replace(enemy[enemy_ids[0]], p=Vector2D(1.0, 3.0))
    enemy[enemy_ids[1]] = dataclasses.replace(enemy[enemy_ids[1]], p=Vector2D(1.0, -3.0))
    _with_frame(game, friendly, enemy, ball_xy=ball_xy)
    return runner.my.game


def test_passer_aims_the_ball_not_its_own_centre_at_the_receiver(runner):
    """The ball leaves from where it sits on the dribbler, along the passer's heading. A
    passer whose centre points exactly at the receiver, with the ball 0.036 m to one side,
    sends it 0.036 m beside the receiver plus whatever heading error the 0.05 rad
    tolerance allows: short passes aimed from the passer's centre were missed with the
    heading error and the ball's offset on the same side in 27 of 29 cases
    (tournament_20260928_221404). Aim from the ball: 3.4 degrees off here, outside the
    tolerance, so the passer turns right instead of kicking."""
    game = _short_pass_with_ball_off_centre(runner, passer_orientation=0.0)

    commands, _pass_complete, _lane_blocked = _pass_exec(game, _ctx(runner), passer_id=1, receiver_id=2)

    assert not commands[1].kick
    assert commands[1].angular_vel < 0.0  # turning right, putting the ball's line onto the receiver


def test_passer_kicks_once_the_ball_line_meets_the_receiver(runner):
    """The same pass with the passer turned so the ball's own line runs through the
    receiver's centre: it kicks."""
    game = _short_pass_with_ball_off_centre(runner, passer_orientation=math.atan2(-0.036, 0.71 - 0.09))

    commands, _pass_complete, _lane_blocked = _pass_exec(game, _ctx(runner), passer_id=1, receiver_id=2)

    assert commands[1].kick


def test_receiver_steps_onto_a_rolling_pass_it_would_meet_off_centre(runner):
    """Once the pass is rolling, the receiver must meet it on the ball's actual path.
    Here it stands 0.07 m beside that path, inside `at_target`'s 0.08 m of the receive
    point, so it used to stop and take the ball on the side of the dribbler: well-faced
    misses met the ball a median 0.08 m off-centre, catches 0.025 m, and passes of
    1.2-2.5 m were caught 45% of the time (tournament_20260924_124033)."""
    game = runner.my.game
    friendly = dict(game.current.friendly_robots)
    enemy = dict(game.current.enemy_robots)

    friendly[1] = dataclasses.replace(friendly[1], has_ball=False, p=Vector2D(0.0, 0.0), orientation=0.0)
    friendly[2] = dataclasses.replace(friendly[2], has_ball=False, p=Vector2D(1.5, 0.07), orientation=math.pi)
    enemy_ids = list(enemy.keys())[:2]
    enemy[enemy_ids[0]] = dataclasses.replace(enemy[enemy_ids[0]], p=Vector2D(1.0, 3.0))
    enemy[enemy_ids[1]] = dataclasses.replace(enemy[enemy_ids[1]], p=Vector2D(1.0, -3.0))

    _with_frame(game, friendly, enemy, ball_xy=(0.4, 0.0), ball_v=(3.0, 0.0))
    game = runner.my.game

    commands, _pass_complete, _lane_blocked = _pass_exec(game, _ctx(runner), passer_id=1, receiver_id=2)

    # Facing -x, the robot's left is -y: towards the ball's path.
    assert commands[2].local_left_vel > 0.05
    assert commands[2].dribble
