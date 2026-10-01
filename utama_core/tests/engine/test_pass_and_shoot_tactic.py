"""Unit-level (no rsim) tests for `PassAndShootTactic`'s assignment/commit logic.

The full end-to-end behaviour (actual robot motion, pass execution, scoring)
is exercised in Utama-Strategy's `test_pass_and_shoot_functional.py`, which
requires `StrategyRunner` + rsim (`mode="rsim"`) — standing that up inside
Utama-Core is out of scope for this pass. What's tested here instead is the
part that doesn't need simulation: `assign_passer_receiver`'s ball-proximity
logic and `is_committed()`'s phase gating, which are plain functions over a
`Game`-shaped object and don't need real motion execution to verify.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics._pass_and_score import PassAndScoreMem, run_setup_phase
from utama_core.tactics.pass_and_shoot import (
    _LANE_BLOCKED_ABANDON_TICKS,
    _PHASE_TIMEOUT_TICKS,
    PassAndShootMem,
    PassAndShootTactic,
    assign_passer_receiver,
)


class _FakeVector2D(Vector2D):
    """Vector2D that also answers `to_2d()`, matching `Vector3D`'s API so
    `game.ball.p.to_2d()` works whether `.p` is 2D or 3D in the real code."""

    def to_2d(self) -> Vector2D:
        return Vector2D(self.x, self.y)


@dataclass
class _FakeRobot:
    p: Vector2D


@dataclass
class _FakeBall:
    p: _FakeVector2D


class _FakeGame:
    def __init__(self, ball_pos: Vector2D, robot_positions: dict[int, Vector2D]):
        self.ball = _FakeBall(p=_FakeVector2D(ball_pos.x, ball_pos.y))
        self.friendly_robots = {rid: _FakeRobot(p=pos) for rid, pos in robot_positions.items()}


def test_assign_passer_receiver_picks_closest_robot_as_passer():
    game = _FakeGame(ball_pos=Vector2D(0, 0), robot_positions={7: Vector2D(1, 0), 12: Vector2D(5, 0)})
    passer, receiver = assign_passer_receiver(game, (7, 12))
    assert passer == 7
    assert receiver == 12


def test_assign_passer_receiver_order_independent_of_input_tuple_order():
    game = _FakeGame(ball_pos=Vector2D(0, 0), robot_positions={7: Vector2D(5, 0), 12: Vector2D(1, 0)})
    passer, receiver = assign_passer_receiver(game, (7, 12))
    assert passer == 12
    assert receiver == 7


def test_assign_passer_receiver_falls_back_when_closest_not_in_pair():
    """A third robot elsewhere on the field is closer to the ball than
    either robot in this tactic's pair — irrelevant, since distances are
    only ever computed over `robot_ids`. Falls back to the first robot_id
    when both robots in the pair are equidistant from the ball."""
    game = _FakeGame(
        ball_pos=Vector2D(0, 0), robot_positions={7: Vector2D(2, 0), 12: Vector2D(2, 0), 99: Vector2D(0.1, 0)}
    )
    passer, receiver = assign_passer_receiver(game, (7, 12))
    assert passer == 7
    assert receiver == 12


def test_assign_passer_receiver_falls_back_when_ball_lookup_empty():
    """Neither robot in the pair is present in `friendly_robots` — falls
    back to `(robot_ids[0], robot_ids[1])`."""
    game = _FakeGame(ball_pos=Vector2D(0, 0), robot_positions={})
    passer, receiver = assign_passer_receiver(game, (7, 12))
    assert passer == 7
    assert receiver == 12


def test_committed_is_false_before_any_assignment():
    tactic = PassAndShootTactic()
    mem = tactic.initial_mem()
    assert tactic.is_committed(game=None, mem=mem) is False


def test_committed_is_false_during_setup_phase():
    tactic = PassAndShootTactic()
    mem = PassAndShootMem(pass_and_score=PassAndScoreMem(phase="setup"), assigned_pair=(1, 2))
    assert tactic.is_committed(game=None, mem=mem) is False


def test_committed_is_true_once_past_setup_phase():
    tactic = PassAndShootTactic()
    mem = PassAndShootMem(pass_and_score=PassAndScoreMem(phase="pass_then_score"), assigned_pair=(1, 2))
    assert tactic.is_committed(game=None, mem=mem) is True

    mem.pass_and_score.phase = "score"
    assert tactic.is_committed(game=None, mem=mem) is True


# ---------------------------------------------------------------------------
# Setup-phase carry (tournament_20260929_163151: two low_block restarts held
# 10s and ending in COMMITTED_FROZEN stalls)
# ---------------------------------------------------------------------------

_PASSER_SPOT = Vector2D(2.0, 1.9)
_RECEIVER_SPOT = Vector2D(2.0, -1.9)


class _RecordingMotionController(MotionController):
    def __init__(self):
        super().__init__(mode="rsim")
        self.targets: dict = {}

    def calculate(self, game, robot_id, target_pos, target_oren):
        self.targets[robot_id] = target_pos
        return Vector2D(0.0, 0.0), 0.0


def _robot(rid: int, pos: Vector2D, friendly: bool = True, has_ball: bool = False) -> Robot:
    return Robot(
        id=rid,
        is_friendly=friendly,
        has_ball=has_ball,
        p=pos,
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _setup_game(
    ball_gap: float, has_ball: bool, enemy_pos: Vector2D = Vector2D(-3.0, 2.5), receiver_pos: Vector2D = _RECEIVER_SPOT
) -> Game:
    """Passer 1 at (-1, 0) facing +x, the ball `ball_gap` in front of its centre;
    receiver 2 already on its spot. `has_ball` is the strict (contact) sensor."""
    friendly = {
        1: _robot(1, Vector2D(-1.0, 0.0), has_ball=has_ball),
        2: _robot(2, receiver_pos),
    }
    enemy = {5: _robot(5, enemy_pos, friendly=False)}
    ball = Ball(Vector3D(-1.0 + ball_gap, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0))
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=False, friendly_robots=friendly, enemy_robots=enemy, ball=ball
    )
    field = Field(
        my_team_is_right=False, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def _setup_mem(carry_origin=None) -> PassAndScoreMem:
    mem = PassAndScoreMem(locked_assignment=(1, 2), passer_position=_PASSER_SPOT, receiver_position=_RECEIVER_SPOT)
    mem.carry_origin = carry_origin
    return mem


def test_setup_carries_a_ball_on_the_dribbler_with_the_dribbler_on():
    """The passer drove to its spot with the dribbler off and left its own ball behind."""
    game = _setup_game(ball_gap=0.1, has_ball=True)
    mc = _RecordingMotionController()
    commands, _ = run_setup_phase(game, TickContext(motion_controller=mc), 1, 2, _setup_mem())
    assert mc.targets[1] == _PASSER_SPOT
    assert commands[1].dribble


def test_setup_closes_the_gap_to_a_ball_only_in_the_visual_box():
    """0.13 m ahead is inside the visual box (0.14 m) but off the dribbler: setting
    off for the spot from there drove away from a ball it never touched."""
    game = _setup_game(ball_gap=0.13, has_ball=False)
    mc = _RecordingMotionController()
    commands, complete = run_setup_phase(game, TickContext(motion_controller=mc), 1, 2, _setup_mem())
    assert mc.targets[1] == game.ball.p.to_2d()
    assert commands[1].dribble
    assert not complete


@pytest.mark.parametrize("carried, spent", [(0.75, False), (0.81, True)])
def test_setup_carry_stops_at_the_shared_carry_limit(carried, spent):
    """passer_position can be metres from the ball; carried there, it was an
    excessive-dribbling foul. At CARRY_LIMIT_M (0.8 m) the passer holds and is set up."""
    game = _setup_game(ball_gap=0.1, has_ball=True)
    origin = Vector2D(game.ball.p.x - carried, 0.0)
    mc = _RecordingMotionController()
    commands, complete = run_setup_phase(game, TickContext(motion_controller=mc), 1, 2, _setup_mem(origin))
    assert complete is spent
    assert (1 not in mc.targets) is spent
    assert commands[1].dribble


def test_a_phase_timeout_keeps_the_carry_and_samples_new_positions():
    """A held ball restarted its carry allowance on every timeout and fouled; and the
    timeout's re-sample was seeded on the pair alone, so it re-picked the same spots."""
    game = _setup_game(ball_gap=0.1, has_ball=True)
    tactic = PassAndShootTactic()
    origin = Vector2D(-0.5, 0.3)
    first = PassAndScoreMem()
    first.carry_origin = origin
    mem = PassAndShootMem(pass_and_score=first, assigned_pair=(1, 2))
    ctx = TickContext(motion_controller=_RecordingMotionController())
    _, mem = tactic.tick(game, ctx, (1, 2), mem)
    before = (mem.pass_and_score.passer_position, mem.pass_and_score.receiver_position)

    mem.pass_and_score.phase_ticks = _PHASE_TIMEOUT_TICKS
    _, mem = tactic.tick(game, ctx, (1, 2), mem)

    assert mem.pass_and_score.carry_origin == origin
    assert (mem.pass_and_score.passer_position, mem.pass_and_score.receiver_position) != before


@pytest.mark.parametrize("carried, kicks", [(0.5, False), (0.81, True)])
def test_a_passer_whose_carry_is_spent_kicks_into_a_blocked_lane(carried, kicks):
    """A blocked lane abandons the pass to re-sample setup spots, which only helps a
    passer that can still carry the ball to one. Holding it at the carry limit:
    - re-sampled, every new lane was blocked too and the ball sat still for 10 s
      (low_block_vs_overload_press, tournament_20260929_171005, 16.3 s);
    - kept aiming instead, it waited for a receiver that never got to its receive
      point, past the referee's 10 s (low_block_vs_switch_of_play,
      tournament_20261001_091050, 38.1 s).
    So it kicks at the receiver as it stands. Here the receiver faces away (not
    ready) and the passer already faces it."""
    game = _setup_game(ball_gap=0.1, has_ball=True, enemy_pos=Vector2D(0.5, 0.0), receiver_pos=Vector2D(2.0, 0.0))
    origin = Vector2D(game.ball.p.x - carried, 0.0)
    inner = _setup_mem(origin)
    inner.phase = "pass_then_score"
    inner.lane_blocked_ticks = _LANE_BLOCKED_ABANDON_TICKS
    mem = PassAndShootMem(pass_and_score=inner, assigned_pair=(1, 2))
    ctx = TickContext(motion_controller=_RecordingMotionController())

    commands, mem = PassAndShootTactic().tick(game, ctx, (1, 2), mem)

    assert bool(commands[1].kick) is kicks
    assert (mem.pass_and_score.phase_ticks > _PHASE_TIMEOUT_TICKS) is not kicks
