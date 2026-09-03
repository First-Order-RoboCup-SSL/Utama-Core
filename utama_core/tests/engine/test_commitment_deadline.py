"""Tests for `Strategy`'s commitment deadline — the kernel-level stall breaker.

Pure scheduling logic, no rsim: mirrors `test_strategy.py`'s bare `_FakeGame`
pattern (the kernel only ever reads `game.referee.referee_command`, `game.ts`,
and `game.ball` off whatever it's given) plus a stub `Tactic` whose
`is_committed()` always returns `True` — the exact "missing release path"
shape behind the 42 "match freeze" bug-fix commits this mechanism responds
to (see `docs/tactic_model_design_decisions.md` §3/§15's "open item").

The deadline overrides `_choose_partition`'s pinning decision only — it does
not change `Tactic.is_committed()` semantics (still `True` throughout every
test here) and does not touch the pinned goalkeeper slot at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.match_log import IntentionEvent, MatchLog
from utama_core.engine.strategy import Strategy
from utama_core.engine.tactic import BaseTactic, TacticTag
from utama_core.entities.referee.referee_command import RefereeCommand


@dataclass
class _FakeReferee:
    referee_command: Optional[RefereeCommand]


@dataclass
class _FakeVector:
    x: float
    y: float

    def distance_to(self, other: "_FakeVector") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class _FakeBall:
    p: _FakeVector


class _FakeGame:
    def __init__(
        self,
        referee_command: Optional[RefereeCommand] = None,
        ts: float = 0.0,
        ball_x: float = 0.0,
        ball_y: float = 0.0,
    ):
        self.referee = _FakeReferee(referee_command) if referee_command is not None else None
        self.ts = ts
        self.ball = _FakeBall(_FakeVector(ball_x, ball_y))


@dataclass
class RecordingMem:
    tick_count: int = 0


class AlwaysCommittedTactic(BaseTactic[RecordingMem]):
    """Stub `Tactic` for a "missing release path" bug: `is_committed()` never
    returns `False` on its own, mirroring the recurring real-world shape
    (a tactic hand-rolls its own timeout, the timeout has a gap, the slot
    never releases and the match freezes for the rest of the game)."""

    tag = TacticTag.MIXED

    def __init__(self):
        self.mem_creations = 0

    def initial_mem(self) -> RecordingMem:
        self.mem_creations += 1
        return RecordingMem()

    def tick(self, game, ctx, robot_ids, mem):
        mem.tick_count += 1
        return {rid: f"cmd-{rid}" for rid in robot_ids}, mem

    def is_committed(self, game, mem) -> bool:
        return True


def _ctx() -> TickContext:
    return TickContext(motion_controller=None)


def _always_a_picker(game, free_robots, prev_partition, applicable_tactic_ids):
    """Always wants to hand the entire free pool to 'a' — the same tactic
    that (before a deadline release) is already pinned there, so a release
    is visible as "the picker got a chance to act on free_robots={1, 2}",
    not as a switch to a different tactic id."""
    return {"a": free_robots} if free_robots else {}


def test_frozen_ball_is_released_after_deadline_with_logged_intention():
    tactic = AlwaysCommittedTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=_always_a_picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
        commitment_deadline_s=10.0,
    )
    strategy.match_log = MatchLog()

    # Tick 1: ts=0, ball at origin -- tactic "a" gets (1, 2) for the first
    # time (no slot exists yet, so is_committed() isn't consulted this tick
    # -- see `Strategy._choose_partition`/`_slot_for`).
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=0.0, ball_x=0.0, ball_y=0.0))
    assert strategy.active_partition == {"a": frozenset({1, 2})}

    # Tick 2: ts=1, ball still at the same spot -- NOW is_committed() is
    # consulted for the first time and the commitment clock starts, anchored
    # at ts=1 / ball=(0, 0).
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=1.0, ball_x=0.0, ball_y=0.0))
    assert strategy._slots["a"].committed_ticks == 1

    # Ball frozen at the exact same spot, well past the 10s deadline
    # (measured from the ts=1 anchor above).
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=12.0, ball_x=0.0, ball_y=0.0))

    # Released: the picker was handed {1, 2} as free and reassigned them
    # straight back to "a" (nothing else wants them) -- robots unchanged as
    # far as the Partitioner is concerned, but the release itself forces a
    # fresh mem (so the tactic can't immediately re-declare committed off
    # the exact stuck state that caused the stall) and restarts the
    # commitment clock (see `_deadline_release`).
    assert strategy.active_partition == {"a": frozenset({1, 2})}
    assert strategy._slots["a"].committed_ticks == 0  # clock restarted (not consulted again until next tick)
    assert tactic.mem_creations == 2  # mem reset by the release itself

    events = [e for e in strategy.match_log.events() if isinstance(e, IntentionEvent)]
    deadline_events = [e for e in events if e.note and "deadline release" in e.note]
    assert len(deadline_events) == 1
    assert deadline_events[0].tactic_id == "a"
    assert "committed" in deadline_events[0].note
    assert "ball moved" in deadline_events[0].note


def test_moving_ball_is_never_released():
    tactic = AlwaysCommittedTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=_always_a_picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
        commitment_deadline_s=10.0,
    )
    strategy.match_log = MatchLog()

    strategy.tick(
        _FakeGame(RefereeCommand.NORMAL_START, ts=0.0, ball_x=0.0, ball_y=0.0)
    )  # slot created, no commitment check yet
    strategy.tick(
        _FakeGame(RefereeCommand.NORMAL_START, ts=1.0, ball_x=0.5, ball_y=0.0)
    )  # commitment clock anchors here
    assert strategy._slots["a"].committed_ticks == 1

    # Ball moves noticeably every tick -- real progress, even well past the
    # nominal deadline in elapsed time.
    for i in range(2, 21):
        strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=float(i), ball_x=float(i) * 0.5, ball_y=0.0))

    # Never released: commitment clock keeps climbing, no reassignment.
    assert strategy.active_partition == {"a": frozenset({1, 2})}
    assert strategy._slots["a"].committed_ticks == 20
    assert tactic.mem_creations == 1  # never reset

    events = [e for e in strategy.match_log.events() if isinstance(e, IntentionEvent)]
    deadline_events = [e for e in events if e.note and "deadline release" in e.note]
    assert deadline_events == []


def test_deadline_none_never_releases():
    tactic = AlwaysCommittedTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=_always_a_picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
        commitment_deadline_s=None,
    )
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=0.0, ball_x=0.0, ball_y=0.0))  # slot created
    # Ball frozen, way past what would otherwise be a deadline.
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=1000.0, ball_x=0.0, ball_y=0.0))
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=2000.0, ball_x=0.0, ball_y=0.0))

    assert strategy.active_partition == {"a": frozenset({1, 2})}
    assert strategy._slots["a"].committed_ticks == 2
    assert tactic.mem_creations == 1  # never reset -- unbounded veto, as before this feature existed

    events = [e for e in strategy.match_log.events() if isinstance(e, IntentionEvent)]
    deadline_events = [e for e in events if e.note and "deadline release" in e.note]
    assert deadline_events == []


def test_release_resets_mem_and_restarts_clock_without_immediate_re_release_loop():
    """After a release, if the Partitioner reassigns the same robots back to
    the same slot, mem must reset and the commitment clock must restart from
    0 -- not remain effectively "still expired," which would otherwise
    re-release every single tick forever (a tight release/reassign loop that
    never lets the tactic actually run)."""
    tactic = AlwaysCommittedTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=_always_a_picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
        commitment_deadline_s=10.0,
    )
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=0.0, ball_x=0.0, ball_y=0.0))  # slot created
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=1.0, ball_x=0.0, ball_y=0.0))  # clock anchors at ts=1
    assert strategy._slots["a"].committed_ticks == 1

    # Just past the deadline (anchored at ts=1) -- triggers exactly one release.
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=11.5, ball_x=0.0, ball_y=0.0))
    assert tactic.mem_creations == 2
    assert strategy._slots["a"].committed_ticks == 0  # released this tick, not yet re-checked

    # Immediately afterward (ts barely advanced, still frozen ball): must NOT
    # release again -- the clock restarted at ts=11.5, so elapsed since then
    # is ~0, nowhere near the 10s deadline.
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=11.6, ball_x=0.0, ball_y=0.0))
    assert tactic.mem_creations == 2  # unchanged -- no second reassignment
    assert strategy._slots["a"].committed_ticks == 1

    events = [e for e in strategy.match_log.events() if isinstance(e, IntentionEvent)]
    deadline_events = [e for e in events if e.note and "deadline release" in e.note]
    assert len(deadline_events) == 1  # exactly one release, not a loop


def test_pinned_goalkeeper_is_never_touched_by_the_deadline():
    tactic = AlwaysCommittedTactic()
    goalkeeper_tactic = AlwaysCommittedTactic()
    strategy = Strategy(
        tactics={"a": tactic},
        partitioner=_always_a_picker,
        outfield_robot_ids=(1, 2),
        ctx=_ctx(),
        commitment_deadline_s=10.0,
    )
    strategy.set_goalkeeper(0, tactic=goalkeeper_tactic)
    strategy.match_log = MatchLog()

    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=0.0, ball_x=0.0, ball_y=0.0))  # slot "a" created
    strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=1.0, ball_x=0.0, ball_y=0.0))  # "a"'s clock anchors
    assert goalkeeper_tactic.mem_creations == 1

    # Ball frozen well past the deadline -- the outfield slot "a" releases,
    # but the pinned goalkeeper slot (never handed to a Partitioner, never
    # part of `_slots`/`committed_ticks` bookkeeping) must be completely
    # unaffected: no new initial_mem(), no intention event naming it, still
    # ticked on robot 0 every tick regardless of the deadline mechanism.
    commands = strategy.tick(_FakeGame(RefereeCommand.NORMAL_START, ts=20.0, ball_x=0.0, ball_y=0.0))
    assert goalkeeper_tactic.mem_creations == 1  # unchanged -- pinned slot never resets via this path
    assert 0 in commands  # goalkeeper still ticked

    events = [e for e in strategy.match_log.events() if isinstance(e, IntentionEvent)]
    assert all(e.tactic_id != "goalkeeper" for e in events)
    deadline_events = [e for e in events if e.note and "deadline release" in e.note]
    assert deadline_events and deadline_events[0].tactic_id == "a"
