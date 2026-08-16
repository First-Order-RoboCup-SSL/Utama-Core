"""Unit-level (no rsim) tests for `TwoRobotAttackTactic`'s assignment/commit logic.

The full end-to-end behaviour (actual robot motion, pass execution, scoring)
is exercised in Utama-Strategy's `test_two_robot_attack_functional.py`, which
requires `StrategyRunner` + rsim (`mode="rsim"`) — standing that up inside
Utama-Core is out of scope for this pass. What's tested here instead is the
part that doesn't need simulation: `assign_passer_receiver`'s ball-proximity
logic and `is_committed()`'s phase gating, which are plain functions over a
`Game`-shaped object and don't need real motion execution to verify.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.entities.data.vector import Vector2D
from utama_core.tactics._pass_and_score import PassAndScoreMem
from utama_core.tactics.two_robot_attack import (
    TwoRobotAttackMem,
    TwoRobotAttackTactic,
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
    tactic = TwoRobotAttackTactic()
    mem = tactic.initial_mem()
    assert tactic.is_committed(game=None, mem=mem) is False


def test_committed_is_false_during_setup_phase():
    tactic = TwoRobotAttackTactic()
    mem = TwoRobotAttackMem(pass_and_score=PassAndScoreMem(phase="setup"), assigned_pair=(1, 2))
    assert tactic.is_committed(game=None, mem=mem) is False


def test_committed_is_true_once_past_setup_phase():
    tactic = TwoRobotAttackTactic()
    mem = TwoRobotAttackMem(pass_and_score=PassAndScoreMem(phase="pass_then_score"), assigned_pair=(1, 2))
    assert tactic.is_committed(game=None, mem=mem) is True

    mem.pass_and_score.phase = "score"
    assert tactic.is_committed(game=None, mem=mem) is True
