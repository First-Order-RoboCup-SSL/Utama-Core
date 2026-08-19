"""Tests for `utama_core.kernel.strategy.Strategy` — pure scheduling logic.

No rsim, no real `Game`/`RobotCommand` machinery: the kernel only reads
`game.referee.referee_command` off whatever object it's given, so a bare
stand-in is enough to exercise the scheduling invariants in isolation.

Covers both shapes `Strategy` supports: the single-active-tactic case (via
`Strategy.single_tactic_picker`, the historical `Picker` API) and genuine
concurrent multi-tactic partitions (via a raw `Partitioner`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel.context import KernelContext
from utama_core.kernel.match_log import MatchLog
from utama_core.kernel.strategy import Strategy
from utama_core.kernel.tactic import BaseTactic, TacticTag


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
    """Counts ticks and initial-mem creations; never commits/always applicable by default."""

    tag = TacticTag.MIXED

    def __init__(self, committed: bool = False, applicable: bool = True):
        self._committed = committed
        self._applicable = applicable
        self.mem_creations = 0
        self.last_robot_ids: tuple[int, ...] = ()

    def initial_mem(self) -> RecordingMem:
        self.mem_creations += 1
        return RecordingMem()

    def tick(self, game, ctx, robot_ids, mem):
        mem.tick_count += 1
        self.last_robot_ids = robot_ids
        return {rid: f"cmd-{rid}" for rid in robot_ids}, mem

    def is_committed(self, game, mem) -> bool:
        return self._committed

    def applicable(self, game) -> bool:
        return self._applicable


def _ctx() -> KernelContext:
    return KernelContext(motion_controller=None)


# --- single-active-tactic shape (Strategy.single_tactic_picker) ---


def test_mem_resets_when_robot_set_changes():
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "a"),
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
        partitioner=Strategy.single_tactic_picker(lambda g, a: "a"),
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
        partitioner=Strategy.single_tactic_picker(lambda game, active: next(picks)),
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
        partitioner=Strategy.single_tactic_picker(lambda game, active: "b"),  # picker always wants "b"
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

    # ...then releases the instant is_committed() flips back to False, and the
    # picker's ("a") choice takes effect on the very next tick.
    strategy._partitioner = Strategy.single_tactic_picker(lambda game, active: "a")
    strategy.tick(_FakeGame())
    assert strategy.active_tactic_id == "a"


def test_barrier_reset_clears_mem_and_overrides_commitment():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    picks = iter(["a", "b", "b"])
    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: next(picks)),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert strategy.active_tactic_id == "a"
    assert committed_tactic.mem_creations == 1

    # Committed — would normally block the picker's desire to switch to "b".
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert strategy.active_tactic_id == "a"

    # Barrier-tier transition: entering GOAL_YELLOW from NORMAL_START. (Not
    # BALL_PLACEMENT/KICKOFF/etc — those are also referee-override commands,
    # see test_override_command_bypasses_tactics_but_still_barrier_resets,
    # and would make the picker's tactic never tick at all here.)
    strategy.tick(_FakeGame(RefereeCommand.GOAL_YELLOW))
    # Barrier reset clears the active tactic and all slot state; the tick
    # after a barrier-entry command choose freshly via the picker again.
    # GOAL_YELLOW itself is not a pause command, so ticking continues.
    assert strategy.active_tactic_id == "b"
    assert other_tactic.mem_creations == 1


def test_pause_command_freezes_without_resetting_mem():
    tactic = RecordingTactic(committed=True)
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "a"),
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
        partitioner=Strategy.single_tactic_picker(lambda game, active: "nonexistent"),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    with pytest.raises(KeyError):
        strategy.tick(_FakeGame())


def test_requires_at_least_one_tactic():
    with pytest.raises(ValueError):
        Strategy(
            tactics={},
            partitioner=Strategy.single_tactic_picker(lambda g, a: "a"),
            outfield_robot_ids=(1, 2),
            ctx=_ctx(),
        )


# --- genuine concurrent multi-tactic partition shape (raw Partitioner) ---


def _even_split_picker(game, free_robots, prev_partition, applicable_tactic_ids):
    """Splits the free pool in half between 'a' and 'b', by sorted id."""
    ordered = sorted(free_robots)
    half = len(ordered) // 2
    return {"a": frozenset(ordered[:half]), "b": frozenset(ordered[half:])}


def test_two_tactics_run_concurrently_each_tick():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=_even_split_picker,
        outfield_robot_ids=(1, 2, 3, 4),
        ctx=_ctx(),
    )
    commands = strategy.tick(_FakeGame())
    assert set(commands.keys()) == {1, 2, 3, 4}
    assert strategy.active_partition == {"a": frozenset({1, 2}), "b": frozenset({3, 4})}


def test_mem_resets_only_for_the_tactic_whose_robots_changed():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()

    def picker(game, free_robots, prev, applicable_tactic_ids):
        return {"a": frozenset({1, 2}), "b": frozenset({3})}

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=picker,
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

    def picker(game, free_robots, prev, applicable_tactic_ids):
        calls.append(free_robots)
        return {"b": free_robots}

    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        partitioner=picker,
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
    strategy._slot_for("a").mem = committed_tactic.initial_mem()

    strategy.tick(_FakeGame())
    # "a" is committed and pinned to {1,2}; picker only ever sees {3,4} now.
    assert calls[-1] == frozenset({3, 4})
    assert strategy.active_partition["a"] == frozenset({1, 2})


def test_picker_assigning_to_a_committed_tactic_raises():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    def picker(game, free_robots, prev, applicable_tactic_ids):
        return {"a": free_robots}  # illegal: "a" is committed and pinned

    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        partitioner=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy._slot_for("a").assigned_robots = frozenset({1, 2})
    strategy._slot_for("a").mem = committed_tactic.initial_mem()

    with pytest.raises(ValueError, match="committed"):
        strategy.tick(_FakeGame())


def test_partition_must_be_exhaustive_and_disjoint():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()

    def missing_robot_picker(game, free_robots, prev, applicable_tactic_ids):
        ordered = sorted(free_robots)
        return {"a": frozenset(ordered[:-1]), "b": frozenset()}  # drops the last robot

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=missing_robot_picker,
        outfield_robot_ids=(1, 2, 3),
        ctx=_ctx(),
    )
    with pytest.raises(ValueError, match="exhaustive"):
        strategy.tick(_FakeGame())


def test_overlapping_partition_raises():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()

    def overlapping_picker(game, free_robots, prev, applicable_tactic_ids):
        return {"a": free_robots, "b": free_robots}  # same robots in both tactics

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=overlapping_picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    with pytest.raises(ValueError, match="more than one tactic"):
        strategy.tick(_FakeGame())


def test_barrier_reset_clears_all_tactics_and_unpins_commitments():
    committed_tactic = RecordingTactic(committed=True)
    other_tactic = RecordingTactic(committed=False)

    calls = []

    def picker(game, free_robots, prev, applicable_tactic_ids):
        calls.append(free_robots)
        return {"b": free_robots}

    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        partitioner=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    # Seed "a" as committed and holding both robots.
    strategy._slot_for("a").assigned_robots = frozenset({1, 2})
    strategy._slot_for("a").mem = committed_tactic.initial_mem()

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    assert calls[-1] == frozenset()  # both robots pinned to "a"

    # Barrier-tier transition clears the pin unconditionally. GOAL_YELLOW, not
    # a referee-override command, so the picker still runs this tick (see
    # test_override_command_bypasses_tactics_but_still_barrier_resets for the
    # override-command case, where the picker is deliberately not called).
    strategy.tick(_FakeGame(RefereeCommand.GOAL_YELLOW))
    assert calls[-1] == frozenset({1, 2})  # "a" no longer holds anything pinned


# --- applicable() filtering (design doc §15) ---


def test_inapplicable_tactic_is_never_proposed_robots():
    """A Partitioner that (incorrectly) proposes robots for an inapplicable,
    non-committed tactic gets a loud ValueError, not a silent bad assignment.
    """
    tactic_a = RecordingTactic(applicable=False)
    tactic_b = RecordingTactic()

    def picker(game, free_robots, prev, applicable_tactic_ids):
        return {"a": free_robots}

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    with pytest.raises(ValueError, match="applicable"):
        strategy.tick(_FakeGame())


def test_inapplicable_tactic_with_no_robots_proposed_does_not_raise():
    """applicable()=False only matters if a Partitioner actually tries to use
    that tactic id — an inapplicable tactic simply never receiving robots is
    fine, not an error condition by itself.
    """
    tactic_a = RecordingTactic(applicable=False)
    tactic_b = RecordingTactic()

    def picker(game, free_robots, prev, applicable_tactic_ids):
        return {"b": free_robots}  # never names "a"

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    commands = strategy.tick(_FakeGame())
    assert set(commands.keys()) == {1, 2}
    assert tactic_a.mem_creations == 0


def test_committed_tactic_is_never_asked_applicable():
    """is_committed() short-circuits before applicable() is even consulted —
    a commitment protects an in-progress action regardless of whether the
    tactic would still call itself applicable if asked (design doc §15).
    """
    committed_tactic = RecordingTactic(committed=True, applicable=False)
    other_tactic = RecordingTactic()

    def picker(game, free_robots, prev, applicable_tactic_ids):
        return {"b": free_robots}

    strategy = Strategy(
        tactics={"a": committed_tactic, "b": other_tactic},
        partitioner=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy._slot_for("a").assigned_robots = frozenset({1, 2})
    strategy._slot_for("a").mem = committed_tactic.initial_mem()

    # Must not raise: "a" is committed, so its applicable()=False is never
    # consulted and it stays pinned exactly as any committed tactic would.
    strategy.tick(_FakeGame())
    assert strategy.active_partition.get("a") == frozenset({1, 2})


def test_tactic_becomes_reassignable_once_commitment_and_applicability_both_allow():
    """Once a commitment ends, applicable() is asked for the first time and
    the tactic is evicted immediately if it says False — no special-casing
    needed for "committed but no longer applicable" (design doc §15).
    """
    tactic_a = RecordingTactic(committed=True, applicable=False)
    tactic_b = RecordingTactic()

    def picker(game, free_robots, prev, applicable_tactic_ids):
        return {"b": free_robots}

    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy._slot_for("a").assigned_robots = frozenset({1, 2})
    strategy._slot_for("a").mem = tactic_a.initial_mem()

    # Still committed: "a" stays pinned even though it's not applicable.
    strategy.tick(_FakeGame())
    assert strategy.active_partition.get("a") == frozenset({1, 2})

    # Commitment ends, but "a" is still inapplicable — picker never names it
    # again ("b" claims everything), so "a" is simply dropped, no error.
    tactic_a._committed = False
    strategy.tick(_FakeGame())
    assert "a" not in strategy.active_partition
    assert strategy.active_partition.get("b") == frozenset({1, 2})


def test_pause_freezes_without_resetting_any_tactic():
    tactic_a = RecordingTactic()

    def picker(game, free_robots, prev, applicable_tactic_ids):
        return {"a": free_robots}

    strategy = Strategy(
        tactics={"a": tactic_a},
        partitioner=picker,
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


# --- match_log intention tracing ---


def test_no_match_log_by_default():
    """match_log is None unless a caller explicitly opts in — tick() must not
    require one."""
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "a"),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    assert strategy.match_log is None
    strategy.tick(_FakeGame())  # must not raise


def test_match_log_records_intention_on_fresh_assignment():
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "a"),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame())

    events = strategy.match_log.events()
    assert len(events) == 1
    assert events[0].tactic_id == "a"
    assert events[0].robot_ids == (1, 2)
    assert events[0].tag == TacticTag.MIXED
    assert events[0].tick == 1


def test_match_log_does_not_repeat_for_unchanged_assignment():
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "a"),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame())
    strategy.tick(_FakeGame())  # same robot set — no new event

    assert len(strategy.match_log.events()) == 1


def test_match_log_records_reassignment_when_active_tactic_switches():
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()
    picks = iter(["a", "b"])
    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=Strategy.single_tactic_picker(lambda game, active: next(picks)),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame())
    strategy.tick(_FakeGame())

    events = strategy.match_log.events()
    # tick 1: "a" gets the pool. tick 2: the picker switches to "b", so "a"
    # is released (robot_ids=()) before "b" picks up the freed pool.
    assert [(e.tactic_id, e.robot_ids) for e in events] == [
        ("a", (1, 2)),
        ("a", ()),
        ("b", (1, 2)),
    ]


def test_match_log_records_tactic_release_without_commitment():
    """A slot dropped before ever committing is distinguishable from one that was."""
    tactic_a, tactic_b = RecordingTactic(), RecordingTactic()
    picks = iter(["a", "b"])
    strategy = Strategy(
        tactics={"a": tactic_a, "b": tactic_b},
        partitioner=Strategy.single_tactic_picker(lambda game, active: next(picks)),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame())
    strategy.tick(_FakeGame())

    release_event = strategy.match_log.events()[1]
    assert release_event.tactic_id == "a"
    assert release_event.robot_ids == ()
    assert "committed=False" in release_event.note
    assert "held 0 committed ticks" in release_event.note


def test_match_log_records_barrier_reset():
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "a"),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START))
    strategy.tick(_FakeGame(RefereeCommand.GOAL_YELLOW))  # barrier-tier transition

    tactic_ids = [e.tactic_id for e in strategy.match_log.events()]
    assert "__barrier_reset__" in tactic_ids


def test_match_log_skips_barrier_reset_event_when_nothing_was_assigned():
    """A barrier reset before any tactic ever held robots has nothing to
    report — must not emit a spurious event."""
    tactic = RecordingTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "a"),
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
    )
    strategy.match_log = MatchLog()

    # First tick ever is GOAL_YELLOW: classify_transition(None, GOAL_YELLOW)
    # is a barrier tier, but no slot has held robots yet.
    strategy.tick(_FakeGame(RefereeCommand.GOAL_YELLOW))

    tactic_ids = [e.tactic_id for e in strategy.match_log.events()]
    assert "__barrier_reset__" not in tactic_ids
