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
from typing import Optional

from utama_core.entities.data.object import TeamType
from utama_core.tactics._pass_and_score import PassAndScoreMem
from utama_core.tactics.two_robot_attack import (
    TwoRobotAttackMem,
    TwoRobotAttackTactic,
    assign_passer_receiver,
)


@dataclass
class _FakeProximityLookup:
    closest_id: Optional[int]
    distance: float = 0.0

    def closest_to_ball(self, team_type_filter: Optional[TeamType] = None):
        if self.closest_id is None:
            return None, 0.0

        class _Key:
            def __init__(self, id_):
                self.id = id_

        return _Key(self.closest_id), self.distance


class _FakeGame:
    def __init__(self, closest_id: Optional[int]):
        self.proximity_lookup = _FakeProximityLookup(closest_id)


def test_assign_passer_receiver_picks_closest_robot_as_passer():
    game = _FakeGame(closest_id=7)
    passer, receiver = assign_passer_receiver(game, (7, 12))
    assert passer == 7
    assert receiver == 12


def test_assign_passer_receiver_order_independent_of_input_tuple_order():
    game = _FakeGame(closest_id=12)
    passer, receiver = assign_passer_receiver(game, (7, 12))
    assert passer == 12
    assert receiver == 7


def test_assign_passer_receiver_falls_back_when_closest_not_in_pair():
    """Closest-to-ball robot isn't one of this tactic's two robots (e.g. a
    third robot elsewhere on the field) — fall back to the first robot_id.
    """
    game = _FakeGame(closest_id=99)
    passer, receiver = assign_passer_receiver(game, (7, 12))
    assert passer == 7
    assert receiver == 12


def test_assign_passer_receiver_falls_back_when_ball_lookup_empty():
    game = _FakeGame(closest_id=None)
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
