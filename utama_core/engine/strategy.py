"""`Strategy` — the per-team-color scheduler that decides which `Tactic`(s) run.

Runs N>=1 Tactics concurrently, each claiming a disjoint slice of the
outfield robot pool (robot 0 / goalkeeper is pinned separately and never
scheduled). The single-writer partition invariant is what makes this safe at
any N: one scheduling pass decides the *entire* partition before any Tactic
runs, so there is never a window where two Tactics could contend for the
same robot — this was true for N=1 and remains true unchanged for N>1.

No bid/fitness scoring, no state machine — a plain, hand-written
`Partitioner` function decides how to split the *free* robot pool (robots no
committed Tactic is currently pinning) across tactic slots each tick, and
this class enforces the invariants around that decision: `is_committed()` is
an absolute per-slot veto, `mem` resets exactly when a slot's assigned robot
set changes (compared as sets, not tuples), and a referee barrier reset
clears everything, in every slot, unconditionally.

The single-active-tactic case (all outfield robots always in one slot) is
not a separate mechanism — it is what you get from a `Partitioner` that only
ever returns one non-empty group. `Strategy.single_tactic_picker` builds
exactly that from a simpler `(Game, Optional[TacticId]) -> TacticId`
function, for callers with only one tactic kind active at a time who would
otherwise have to write a trivial one-group `Partitioner` themselves.

Referee restarts (kickoff/ball-placement/free-kick/penalty) are handled
before any tactic ticks at all, not by a tactic: `kernel.referee_override`
reuses the BT path's existing `actions.py` Step classes (keep-out-distance
geometry, formation positions) to take over every outfield robot's command
for the duration of the restart, the same way `build_referee_override_tree`
does for the BT path. See `_OVERRIDE_COMMANDS` there and `referee_reset.py`
for how this fits with the barrier-reset/pause tiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from utama_core.engine.context import KernelContext
from utama_core.engine.match_log import MatchLog
from utama_core.engine.referee_override import (
    RefereeActionOverride,
    RefereeOverride,
    is_override_command,
)
from utama_core.engine.referee_reset import ResetTier, classify_transition, is_paused
from utama_core.engine.tactic import RobotId, Tactic, TacticId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand

logger = logging.getLogger(__name__)

# A Partitioner partitions the *free* outfield pool (robots not currently
# pinned by a committed tactic slot — see `Strategy._choose_partition`) into
# named tactic slots every tick, given the game state, the free robot pool,
# the previous full partition (None on the first tick or right after a
# barrier reset), and the set of tactic ids currently applicable() (design
# doc §15) — a tactic id absent from this set must not be given any robots
# this tick, whether because it isn't registered or because its applicable()
# just returned False. It must return a partition that exactly covers
# `free_robot_ids` — every free robot in exactly one slot, no slot given a
# robot outside that pool, and no non-empty slot for a tactic id outside
# `applicable_tactic_ids`. `Strategy.tick()` raises if it doesn't. Committed
# slots are never passed to the partitioner as available; it only ever
# decides what happens to the robots nobody has vetoed keeping.
Partitioner = Callable[
    [Game, frozenset[RobotId], Optional[dict[TacticId, frozenset[RobotId]]], frozenset[TacticId]],
    dict[TacticId, frozenset[RobotId]],
]

# The original single-tactic picker shape: decide which one TacticId should
# own the *entire* outfield pool this tick, given the currently active one
# (None on the first tick or right after a reset). Adapted into a
# `Partitioner` by `Strategy.single_tactic_picker`.
Picker = Callable[[Game, Optional[TacticId]], TacticId]


@dataclass
class _TacticSlot:
    tactic: Tactic
    mem: object = None
    assigned_robots: frozenset[RobotId] = field(default_factory=frozenset)
    committed_ticks: int = 0


class Strategy:
    """Runs one Tactic per slot, with each slot's robot pool decided fresh each tick.

    `is_committed()`/`mem`-reset semantics are per-slot, not pool-wide: one
    slot being committed only blocks *that slot's* robots from being
    reassigned; it has no bearing on any other slot. This is the direct
    consequence of preserving the single-writer invariant per-slot instead of
    per-pool. A barrier reset still clears every slot unconditionally — a
    referee restart makes every slot's in-progress action moot at once, not
    just one.
    """

    def __init__(
        self,
        tactics: dict[TacticId, Tactic],
        partitioner: Partitioner,
        outfield_robot_ids: tuple[RobotId, ...],
        ctx: KernelContext,
        referee_overrides: Optional[dict[RefereeCommand, RefereeActionOverride]] = None,
    ):
        if not tactics:
            raise ValueError("Strategy needs at least one registered tactic")
        self._tactics = dict(tactics)
        self._partitioner = partitioner
        self._outfield_robot_ids = frozenset(outfield_robot_ids)
        self._ctx = ctx

        self._slots: dict[TacticId, _TacticSlot] = {}
        self._prev_partition: Optional[dict[TacticId, frozenset[RobotId]]] = None
        self._prev_referee_command = None
        self._referee_override = RefereeOverride(overrides=referee_overrides)

        # Optional structured intention/trace log — assigned post-construction
        # by `StrategyRunner` (see its `match_log_path` param), not threaded
        # through every `build_*_kernel_strategy` factory's constructor
        # signature, since none of them currently take anything beyond
        # `motion_controller`. `None` disables it entirely. The setter below
        # also pushes it into `self._ctx.match_log`, the one instance shared
        # by every tactic/skill invoked this tick, so assigning it here is
        # the only place a caller ever needs to touch.
        self._match_log: Optional[MatchLog] = None
        self._tick_count = 0

    @property
    def referee_overrides(self) -> dict[RefereeCommand, RefereeActionOverride]:
        return self._referee_override.overrides

    @referee_overrides.setter
    def referee_overrides(self, value: dict[RefereeCommand, RefereeActionOverride]) -> None:
        """Replace the strategy-supplied restart overrides on the live `RefereeOverride`.

        Assigned post-construction the same way `match_log` is — `AbstractStrategy.__init__`
        runs before `load_motion_controller` builds the kernel `Strategy` via each
        `build_kernel_strategy(motion_controller)` factory, none of which know about
        `referee_overrides`, so `AbstractStrategy.load_motion_controller` sets this right
        after construction instead of threading a new constructor kwarg through every
        existing factory.
        """
        self._referee_override.overrides = value

    @property
    def match_log(self) -> Optional[MatchLog]:
        return self._match_log

    @match_log.setter
    def match_log(self, value: Optional[MatchLog]) -> None:
        self._match_log = value
        self._ctx.match_log = value

    @staticmethod
    def single_tactic_picker(picker: Picker) -> Partitioner:
        """Adapt a single-tactic `Picker` into a `Partitioner` for `Strategy`.

        The adapted picker always assigns the *entire* free pool to whichever
        `TacticId` the wrapped `picker` chooses — reproducing the original
        single-active-tactic behaviour exactly. "Currently active" is read
        from `prev_partition` (the previous tick's full partition, which
        `Strategy` always passes in) rather than kept as separate state
        inside this closure — `Strategy` is the single source of truth for
        what was active, so this stays correct even if a caller swaps in a
        new picker mid-run.

        When the active tactic is committed, the whole outfield pool is
        pinned to it and the free pool handed to this picker is empty; in
        that case the wrapped `picker` is not called at all (mirroring the
        original `Strategy`, which never consulted the picker while the
        active tactic was committed).

        Does not filter by `applicable_tactic_ids` itself — it can't
        distinguish "the wrapped picker named an unregistered tactic" (a
        bug, must raise `KeyError` same as ever) from "it named a
        registered but currently inapplicable one" (`applicable_tactic_ids`
        alone can't tell those apart, since it's already a subset of
        registered ids by construction). `Strategy._choose_partition`'s own
        validation already raises the right error for both cases, so this
        wrapper is deliberately naive and lets that validation decide.
        """

        def _partitioner(
            game: Game,
            free_robots: frozenset[RobotId],
            prev_partition: Optional[dict[TacticId, frozenset[RobotId]]],
            applicable_tactic_ids: frozenset[TacticId],
        ) -> dict[TacticId, frozenset[RobotId]]:
            if not free_robots:
                return {}
            prev_active_id = next(iter(prev_partition), None) if prev_partition else None
            active_id = picker(game, prev_active_id)
            return {active_id: free_robots}

        return _partitioner

    @property
    def active_partition(self) -> dict[TacticId, frozenset[RobotId]]:
        return {tid: slot.assigned_robots for tid, slot in self._slots.items() if slot.assigned_robots}

    def slot_status(self, game: Game) -> dict[TacticId, dict]:
        """Per-active-slot debug info: robots held and whether it's currently committed.

        Read-only reporting, not used by `tick()` itself — for callers that
        want to display "what is this robot's tactic doing right now"
        without reaching into `_slots` directly (e.g. a GUI debug panel).
        """
        return {
            tid: {"robots": slot.assigned_robots, "committed": slot.tactic.is_committed(game, slot.mem)}
            for tid, slot in self._slots.items()
            if slot.assigned_robots
        }

    @property
    def active_tactic_id(self) -> Optional[TacticId]:
        """The single occupied slot's id, for single-tactic-shaped callers.

        Only meaningful when at most one slot ever holds robots at a time
        (e.g. a `Strategy` built via `single_tactic_picker`). With a real
        multi-slot partition this returns whichever slot happened to be
        checked as non-empty first — multi-tactic callers should use
        `active_partition` instead.
        """
        for tid, slot in self._slots.items():
            if slot.assigned_robots:
                return tid
        return None

    def tick(self, game: Game) -> dict[RobotId, RobotCommand]:
        self._tick_count += 1
        referee = getattr(game, "referee", None)
        current_command = getattr(referee, "referee_command", None) if referee is not None else None

        if current_command is not None:
            tier = classify_transition(self._prev_referee_command, current_command)
            if tier is ResetTier.BARRIER:
                self._barrier_reset(game)
            self._prev_referee_command = current_command

            if is_override_command(current_command):
                # Restart in progress (kickoff/placement/free-kick/penalty),
                # or STOP/TIMEOUT_* (see `referee_override.py`'s module
                # docstring for why both are in this set too, ahead of the
                # `is_paused` check below rather than behind it): legal
                # positioning takes over every outfield robot for this tick,
                # same as the BT path's RefereeOverride Selector. Slots
                # already had their mem/commitments cleared by the barrier
                # reset on the transition in; tactics simply don't tick while
                # this is active, so there is nothing further to reconcile
                # once the restart ends and normal picking resumes.
                return self._referee_override.tick(game, self._ctx.motion_controller, current_command)

            if is_paused(current_command):
                # HALT: mem and commitments survive untouched, but no tactic
                # issues motion commands while play is stopped.
                return {}

        partition = self._choose_partition(game)
        self._validate_partition(partition)

        # Any existing slot not mentioned in this tick's partition has lost
        # every robot it had (the picker gave its robots to someone else, or
        # simply stopped naming it) — clear it explicitly. `partition` only
        # ever contains pinned/committed slots plus whatever the picker named
        # this tick, so a slot's absence is itself the "you now have zero
        # robots" signal, not something the main loop below would otherwise see.
        for tactic_id, slot in self._slots.items():
            if tactic_id not in partition and slot.assigned_robots:
                if self.match_log is not None:
                    self.match_log.intention(
                        tick=self._tick_count,
                        sim_time=getattr(game, "ts", 0.0),
                        tactic_id=tactic_id,
                        robot_ids=(),
                        tag=slot.tactic.tag,
                        note=f"released (was committed={slot.committed_ticks > 0}, held {slot.committed_ticks} committed ticks)",
                    )
                slot.mem = None
                slot.assigned_robots = frozenset()
                slot.committed_ticks = 0

        commands: dict[RobotId, RobotCommand] = {}
        for tactic_id, robot_ids in partition.items():
            slot = self._slot_for(tactic_id)

            if slot.assigned_robots != robot_ids:
                slot.mem = slot.tactic.initial_mem() if robot_ids else None
                slot.assigned_robots = robot_ids
                slot.committed_ticks = 0
                if self.match_log is not None and robot_ids:
                    self.match_log.intention(
                        tick=self._tick_count,
                        sim_time=getattr(game, "ts", 0.0),
                        tactic_id=tactic_id,
                        robot_ids=robot_ids,
                        tag=slot.tactic.tag,
                    )

            if not robot_ids:
                # A slot with no robots this tick has nothing to tick —
                # ticking a tactic with an empty robot set and no mem is
                # meaningless, not just a degenerate case of the normal path.
                continue

            ordered_robots = tuple(sorted(robot_ids))
            slot_commands, slot.mem = slot.tactic.tick(game, self._ctx, ordered_robots, slot.mem)
            commands.update(slot_commands)

        self._prev_partition = partition
        return commands

    def _slot_for(self, tactic_id: TacticId) -> _TacticSlot:
        if tactic_id not in self._tactics:
            raise KeyError(f"picker chose unregistered tactic id {tactic_id!r}")
        if tactic_id not in self._slots:
            self._slots[tactic_id] = _TacticSlot(tactic=self._tactics[tactic_id])
        return self._slots[tactic_id]

    def _choose_partition(self, game: Game) -> dict[TacticId, frozenset[RobotId]]:
        """Pin any committed slot's robots, then ask the picker to partition the rest.

        A slot whose active Tactic is `is_committed()` keeps exactly the
        robot set it already has — the picker only ever sees the robots
        nobody has vetoed keeping, mirroring the absolute per-tactic veto
        (design doc §3), now scoped to one slot instead of always the whole
        pool.

        A tactic that is not currently committed is additionally filtered
        out of the picker's candidate set entirely when `applicable(game)`
        is False (design doc §15) — a precondition on being assigned at all,
        checked only for non-committed tactics, never overriding a
        commitment. `applicable_tactic_ids` is passed to the picker so it
        can respect this itself; `Strategy` also validates the picker's
        return value against it afterward, since a `Partitioner` is a plain
        function and nothing stops one from ignoring its own inputs.
        """
        pinned: dict[TacticId, frozenset[RobotId]] = {}
        for tactic_id, slot in self._slots.items():
            if not slot.assigned_robots:
                continue
            if slot.tactic.is_committed(game, slot.mem):
                slot.committed_ticks += 1
                if slot.committed_ticks % 100 == 0:
                    logger.warning(
                        "tactic %r has blocked reassignment for %d consecutive ticks — "
                        "if this keeps climbing, its is_committed() logic is likely stuck",
                        tactic_id,
                        slot.committed_ticks,
                    )
                pinned[tactic_id] = slot.assigned_robots
            else:
                slot.committed_ticks = 0

        pinned_robots = frozenset().union(*pinned.values()) if pinned else frozenset()
        free_robots = self._outfield_robot_ids - pinned_robots

        applicable_tactic_ids = {
            tactic_id
            for tactic_id, tactic in self._tactics.items()
            if tactic_id not in pinned and tactic.applicable(game)
        }

        free_partition = self._partitioner(game, free_robots, self._prev_partition, frozenset(applicable_tactic_ids))

        result = dict(pinned)
        for tactic_id, robots in free_partition.items():
            if tactic_id in pinned:
                raise ValueError(
                    f"picker assigned robots to tactic {tactic_id!r}, which is currently committed "
                    "and must not be reassigned"
                )
            if robots and tactic_id in self._tactics and tactic_id not in applicable_tactic_ids:
                raise ValueError(
                    f"picker assigned robots to tactic {tactic_id!r}, which is not currently "
                    "applicable() — a Partitioner must not propose robots for an inapplicable tactic"
                )
            result[tactic_id] = robots
        return result

    def _validate_partition(self, partition: dict[TacticId, frozenset[RobotId]]) -> None:
        seen: set[RobotId] = set()
        for tactic_id, robots in partition.items():
            if tactic_id not in self._tactics:
                raise KeyError(f"picker chose unregistered tactic id {tactic_id!r}")
            overlap = seen & robots
            if overlap:
                raise ValueError(f"picker assigned robot(s) {sorted(overlap)} to more than one tactic in the same tick")
            seen |= robots

        if seen != self._outfield_robot_ids:
            missing = self._outfield_robot_ids - seen
            extra = seen - self._outfield_robot_ids
            raise ValueError(
                "picker's partition is not an exhaustive, exact cover of the outfield pool "
                f"(missing={sorted(missing)}, unexpected={sorted(extra)})"
            )

    def _barrier_reset(self, game: Game) -> None:
        had_any_assignment = any(slot.assigned_robots for slot in self._slots.values())
        for slot in self._slots.values():
            slot.mem = None
            slot.assigned_robots = frozenset()
            slot.committed_ticks = 0
        self._prev_partition = None
        if self.match_log is not None and had_any_assignment:
            self.match_log.intention(
                tick=self._tick_count,
                sim_time=getattr(game, "ts", 0.0),
                tactic_id="__barrier_reset__",
                robot_ids=(),
                tag=TacticTag.MIXED,
            )
