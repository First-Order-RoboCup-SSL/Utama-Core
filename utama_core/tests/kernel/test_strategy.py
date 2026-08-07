"""Tests for `utama_core.kernel.strategy.Strategy` — pure scheduling logic.

No rsim, no real `Game`/`RobotCommand` machinery: the kernel only reads
`game.referee.referee_command` off whatever object it's given, so a bare
stand-in is enough to exercise the scheduling invariants in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel.context import KernelContext
from utama_core.kernel.strategy import Strategy
from utama_core.kernel.tactic import BaseTactic


@dataclass
class _FakeReferee:
    referee_command: Optional[RefereeCommand]


class _FakeGame:
    def __init__(self, referee_command: Optional[RefereeCommand] = None):
        self.referee = _FakeReferee(referee_command) if referee_command is not None else None


@dataclass
class RecordingMem:
    tick_count: int = 0


class RecordingTactic(BaseTactic[RecordingMem]):
    """Counts ticks and initial-mem creations; never commits by default."""

    def __init__(self, committed: bool = False):
        self._committed = committed
        self.mem_creations = 0

    def make_initial_mem(self) -> RecordingMem:
        self.mem_creations += 1
        return RecordingMem()

    def tick(self, game, ctx, robot_ids, mem):
        mem.tick_count += 1
        return {rid: f"cmd-{rid}" for rid in robot_ids}, mem

    def committed(self, game, mem) -> bool:
        return self._committed


def _ctx() -> KernelContext:
    return KernelContext(motion_controller=None, rsim_env=None)


def test_mem_resets_when_robot_set_changes():
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        picker=lambda game, active: "a",
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.tick(_FakeGame())
    assert tactic.mem_creations == 1

    # Same robot set again next tick: no reset.
    strategy.tick(_FakeGame())
    assert tactic.mem_creations == 1


def test_mem_reset_uses_set_comparison_not_order():
    """The frozenset-vs-tuple bug identified during design: comparing assigned
    robot sets as tuples/lists is order-sensitive and would falsely trigger a
    reset when the same robots are handed back in a different order. The
    kernel stores/compares assignment as a frozenset specifically to rule
    this out — verified directly against the slot's stored assignment.
    """
    tactic = RecordingTactic()
    strategy = Strategy(tactics={"a": tactic}, picker=lambda g, a: "a", outfield_robot_ids=(2, 1), ctx=_ctx())

    strategy.tick(_FakeGame())
    assert tactic.mem_creations == 1
    # Stored assignment is a frozenset, so (2, 1) and (1, 2) are indistinguishable.
    assert strategy._slots["a"].assigned_robots == frozenset({1, 2})

    # Ticking again with the identical outfield tuple must not re-trigger a reset.
    strategy.tick(_FakeGame())
    assert tactic.mem_creations == 1


def test_committed_tactic_blocks_reassignment():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    picks = iter(["a", "b", "b", "b"])  # picker would prefer to switch to "b" after tick 1
    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        picker=lambda game, active: next(picks),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )

    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "a"

    # Picker wants "b", but "a" is committed — must stay on "a".
    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "a"
    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "a"

    assert other_tactic.mem_creations == 0  # never activated


def test_committed_tactic_releases_once_it_stops_committing():
    """A tactic that WAS committed but no longer is must be reassignable again
    the very next tick — the veto is per-tick, not sticky once released.
    """
    tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)
    strategy = Strategy(
        tactics={"a": tactic, "b": other_tactic},
        picker=lambda game, active: "b",  # picker always wants "b"
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )

    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "b"  # nothing active yet on tick 1, picker's choice applies directly

    # Make "b" the incumbent and committed, then confirm the veto holds...
    other_tactic._committed = True
    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "b"
    other_tactic._committed = False

    # ...then releases the instant committed() flips back to False, and the
    # picker's ("a") choice takes effect on the very next tick.
    picker_choice = "a"
    strategy._picker = lambda game, active: picker_choice
    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "a"


def test_barrier_reset_clears_mem_and_overrides_commitment():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    picks = iter(["a", "b", "b"])
    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        picker=lambda game, active: next(picks),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert strategy.active_tactic_id == "a"
    assert committed_tactic.mem_creations == 1

    # Committed — would normally block the picker's desire to switch to "b".
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert strategy.active_tactic_id == "a"

    # Barrier-tier transition: entering BALL_PLACEMENT_YELLOW from NORMAL_START.
    strategy.tick(_FakeGame(RefereeCommand.BALL_PLACEMENT_YELLOW))
    # Barrier reset clears the active tactic and all slot state; the tick
    # after a barrier-entry command choose freshly via the picker again.
    # BALL_PLACEMENT itself is not a pause command, so ticking continues.
    assert strategy.active_tactic_id == "b"
    assert other_tactic.mem_creations == 1


def test_pause_command_freezes_without_resetting_mem():
    tactic = RecordingTactic(committed=True)
    strategy = Strategy(
        tactics={"a": tactic},
        picker=lambda game, active: "a",
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert tactic.mem_creations == 1
    commands = strategy.tick(_FakeGame(RefereeCommand.STOP))
    assert commands == {}  # no commands issued while paused
    assert tactic.mem_creations == 1  # mem NOT reset by a plain pause

    # Resume with FORCE_START coming from STOP (not from a barrier command):
    # should NOT be treated as a barrier reset.
    strategy.tick(_FakeGame(RefereeCommand.FORCE_START))
    assert tactic.mem_creations == 1


def test_unregistered_picker_choice_raises():
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        picker=lambda game, active: "nonexistent",
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    with pytest.raises(KeyError):
        strategy.tick(_FakeGame())


def test_requires_at_least_one_tactic():
    with pytest.raises(ValueError):
        Strategy(tactics={}, picker=lambda g, a: "a", outfield_robot_ids=(1, 2), ctx=_ctx())
