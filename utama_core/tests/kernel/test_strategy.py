"""Tests for `utama_core.kernel.strategy.Strategy` — pure scheduling logic.

No rsim, no real `Game`/`RobotCommand` machinery: the kernel only reads
`game.referee.referee_command` off whatever object it's given, so a bare
stand-in is enough to exercise the scheduling invariants in isolation.

Covers both shapes `Strategy` supports: the single-active-tactic case (via
`Strategy.single_tactic_picker`, the historical `Picker` API) and genuine
concurrent multi-tactic partitions (via a raw `GroupPicker`).
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
        self.last_robot_ids: tuple[int, ...] = ()

    def make_initial_mem(self) -> RecordingMem:
        self.mem_creations += 1
        return RecordingMem()

    def tick(self, game, ctx, robot_ids, mem):
        mem.tick_count += 1
        self.last_robot_ids = robot_ids
        return {rid: f"cmd-{rid}" for rid in robot_ids}, mem

    def committed(self, game, mem) -> bool:
        return self._committed


def _ctx() -> KernelContext:
    return KernelContext(motion_controller=None, rsim_env=None)


# --- single-active-tactic shape (Strategy.single_tactic_picker) ---


def test_mem_resets_when_robot_set_changes():
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        group_picker=Strategy.single_tactic_picker(lambda game, active: "a"),
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
    strategy = Strategy(
        tactics={"a": tactic},
        group_picker=Strategy.single_tactic_picker(lambda g, a: "a"),
        outfield_robot_ids=(2, 1),
        ctx=_ctx(),
    )

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
        group_picker=Strategy.single_tactic_picker(lambda game, active: next(picks)),
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
        group_picker=Strategy.single_tactic_picker(lambda game, active: "b"),  # picker always wants "b"
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
    strategy._group_picker = Strategy.single_tactic_picker(lambda game, active: "a")
    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "a"


def test_barrier_reset_clears_mem_and_overrides_commitment():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    picks = iter(["a", "b", "b"])
    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        group_picker=Strategy.single_tactic_picker(lambda game, active: next(picks)),
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
        group_picker=Strategy.single_tactic_picker(lambda game, active: "a"),
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
        group_picker=Strategy.single_tactic_picker(lambda game, active: "nonexistent"),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    with pytest.raises(KeyError):
        strategy.tick(_FakeGame())


def test_requires_at_least_one_tactic():
    with pytest.raises(ValueError):
        Strategy(
            tactics={},
            group_picker=Strategy.single_tactic_picker(lambda g, a: "a"),
            outfield_robot_ids=(1, 2),
            ctx=_ctx(),
        )


# --- genuine concurrent multi-tactic partition shape (raw GroupPicker) ---


def _even_split_picker(game, free_robots, prev_partition):
    """Splits the free pool in half between 'a' and 'b', by sorted id."""
    ordered = sorted(free_robots)
    half = len(ordered) // 2
    return {"a": frozenset(ordered[:half]), "b": frozenset(ordered[half:])}


def test_two_tactics_run_concurrently_each_tick():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        group_picker=_even_split_picker,
        outfield_robot_ids=(1, 2, 3, 4),
        ctx=_ctx(),
    )
    commands = strategy.tick(_FakeGame())
    assert set(commands.keys()) == {1, 2, 3, 4}
    assert strategy.active_partition == {"a": frozenset({1, 2}), "b": frozenset({3, 4})}


def test_mem_resets_only_for_the_tactic_whose_robots_changed():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()

    def picker(game, free_robots, prev):
        return {"a": frozenset({1, 2}), "b": frozenset({3})}

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        group_picker=picker,
        outfield_robot_ids=(1, 2, 3),
        ctx=_ctx(),
    )
    strategy.tick(_FakeGame())
    assert tactic_a.mem_creations == 1
    assert tactic_b.mem_creations == 1

    strategy.tick(_FakeGame())
    assert tactic_a.mem_creations == 1
    assert tactic_b.mem_creations == 1


def test_committed_group_keeps_its_robots_while_other_group_still_reassigns():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    calls = []

    def picker(game, free_robots, prev):
        calls.append(free_robots)
        return {"b": free_robots}

    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        group_picker=picker,
        outfield_robot_ids=(1, 2, 3, 4),
        ctx=_ctx(),
    )

    # Tick 1: no slot has robots yet, so nothing is pinned — picker sees the
    # full pool and (per its own logic) assigns everything to "b". Tactic "a"
    # has no robots and is skipped entirely this tick.
    strategy.tick(_FakeGame())
    assert strategy.active_partition == {"b": frozenset({1, 2, 3, 4})}

    # Manually seed tactic "a" with robots {1,2} to simulate a prior tick that
    # gave it robots (bypassing the picker, since this test only cares about
    # the pinning behaviour once "a" already holds robots and commits).
    strategy._slot_for("a").assigned_robots = frozenset({1, 2})
    strategy._slot_for("a").mem = committed_tactic.make_initial_mem()

    strategy.tick(_FakeGame())
    # "a" is committed and pinned to {1,2}; picker only ever sees {3,4} now.
    assert calls[-1] == frozenset({3, 4})
    assert strategy.active_partition["a"] == frozenset({1, 2})


def test_picker_assigning_to_a_committed_tactic_raises():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    def picker(game, free_robots, prev):
        return {"a": free_robots}  # illegal: "a" is committed and pinned

    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        group_picker=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy._slot_for("a").assigned_robots = frozenset({1, 2})
    strategy._slot_for("a").mem = committed_tactic.make_initial_mem()

    with pytest.raises(ValueError, match="committed"):
        strategy.tick(_FakeGame())


def test_partition_must_be_exhaustive_and_disjoint():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()

    def missing_robot_picker(game, free_robots, prev):
        ordered = sorted(free_robots)
        return {"a": frozenset(ordered[:-1]), "b": frozenset()}  # drops the last robot

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        group_picker=missing_robot_picker,
        outfield_robot_ids=(1, 2, 3),
        ctx=_ctx(),
    )
    with pytest.raises(ValueError, match="exhaustive"):
        strategy.tick(_FakeGame())


def test_overlapping_partition_raises():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()

    def overlapping_picker(game, free_robots, prev):
        return {"a": free_robots, "b": free_robots}  # same robots in both tactics

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        group_picker=overlapping_picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    with pytest.raises(ValueError, match="more than one tactic"):
        strategy.tick(_FakeGame())


def test_barrier_reset_clears_all_tactics_and_unpins_commitments():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    calls = []

    def picker(game, free_robots, prev):
        calls.append(free_robots)
        return {"b": free_robots}

    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        group_picker=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    # Seed "a" as committed and holding both robots.
    strategy._slot_for("a").assigned_robots = frozenset({1, 2})
    strategy._slot_for("a").mem = committed_tactic.make_initial_mem()

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert calls[-1] == frozenset()  # both robots pinned to "a"

    # Barrier-tier transition clears the pin unconditionally.
    strategy.tick(_FakeGame(RefereeCommand.BALL_PLACEMENT_YELLOW))
    assert calls[-1] == frozenset({1, 2})  # "a" no longer holds anything pinned


def test_pause_freezes_without_resetting_any_tactic():
    tactic_a = RecordingTactic()

    def picker(game, free_robots, prev):
        return {"a": free_robots}

    strategy = Strategy(
        tactics={"a": tactic_a},
        group_picker=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert tactic_a.mem_creations == 1

    commands = strategy.tick(_FakeGame(RefereeCommand.STOP))
    assert commands == {}
    assert tactic_a.mem_creations == 1

    strategy.tick(_FakeGame(RefereeCommand.FORCE_START))
    assert tactic_a.mem_creations == 1
