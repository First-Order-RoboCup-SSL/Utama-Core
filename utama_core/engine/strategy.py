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

from utama_core.engine.context import TickContext
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
from utama_core.tactics.goalkeeper import GoalkeeperTactic

logger = logging.getLogger(__name__)

# Default `commitment_deadline_s` (see `Strategy.__init__`): long enough that
# no legitimate multi-phase tactic (a pass, a relay, a dribble-and-shoot
# sequence) trips it under normal play, short enough that a genuinely stuck
# commitment is a bounded hiccup rather than costing the rest of the match —
# see the 566s frozen-match case in the design discussion this mechanism
# responds to. Not derived from any single tactic's own timeout constants
# (`hop_ticks`/`ticks_held`/etc.) on purpose: this is a kernel-level backstop
# for when those per-tactic timeouts fail to fire, not a tuned replacement
# for any one of them.
DEFAULT_COMMITMENT_DEADLINE_S = 15.0

# A commitment is only treated as "stalled" (eligible for deadline release)
# if the ball has moved less than this since the commitment began. A
# commitment making real progress -- passing, dribbling toward goal -- keeps
# moving the ball well past this within the deadline window; this exists
# purely to distinguish "stuck" from "still working," not to cap how long a
# genuinely progressing play may run.
_STALL_BALL_MOVEMENT_M = 0.05

# A Partitioner partitions the *free* outfield pool (robots not currently
# pinned by a committed tactic slot — see `Strategy._choose_partition`) into
# named tactic slots every tick, given the game state, the free robot pool,
# the previous full partition (None on the first tick or right after a
# barrier reset), and the set of tactic ids currently applicable() (design
# doc §15) — a tactic id absent from this set must not be given any robots
# this tick, whether because it isn't registered or because its applicable()
# just returned False. Every free robot must end up in at most one slot (no
# robot given to two slots at once — `Strategy._validate_partition` raises if
# so) and no slot may be given a robot outside `free_robot_ids` or a
# non-empty slot for a tactic id outside `applicable_tactic_ids`. A free
# robot need NOT appear in the returned partition at all — a Partitioner has
# no obligation to invent an applicable tactic for a robot nothing currently
# wants; an uncovered robot simply isn't ticked by any tactic that tick (see
# `Strategy._validate_partition`'s docstring). Committed slots are never
# passed to the partitioner as available; it only ever decides what happens
# to the robots nobody has vetoed keeping.
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
    # Ball position and sim-time recorded the tick a commitment *began*
    # (committed_ticks went 0 -> 1) — the reference point the commitment
    # deadline (see `Strategy.__init__`'s `commitment_deadline_s`) measures
    # both elapsed time and ball movement against. `None` whenever the slot
    # isn't currently mid-commitment; reset alongside `committed_ticks`
    # everywhere that field is zeroed (release, barrier reset, slot
    # reassignment), so it never outlives the commitment it describes.
    commit_started_ts: Optional[float] = None
    commit_started_ball_pos: Optional[object] = None


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
        ctx: TickContext,
        referee_overrides: Optional[dict[RefereeCommand, RefereeActionOverride]] = None,
        commitment_deadline_s: Optional[float] = DEFAULT_COMMITMENT_DEADLINE_S,
    ):
        if not tactics:
            raise ValueError("Strategy needs at least one registered tactic")
        self._tactics = dict(tactics)
        self._partitioner = partitioner
        self._outfield_robot_ids = frozenset(outfield_robot_ids)
        self._ctx = ctx
        # Kernel-level stall breaker, not a play-length cap — see
        # `_choose_partition`'s deadline check for the actual mechanism and
        # the module-level docstring/`DEFAULT_COMMITMENT_DEADLINE_S` for the
        # rationale. `None` disables it entirely (pre-existing unbounded-veto
        # behaviour, design doc §3).
        self._commitment_deadline_s = commitment_deadline_s

        self._slots: dict[TacticId, _TacticSlot] = {}
        self._prev_partition: Optional[dict[TacticId, frozenset[RobotId]]] = None
        self._prev_referee_command = None
        self._pending_barrier_reset: Optional[dict[TacticId, frozenset[RobotId]]] = None
        self._referee_override = RefereeOverride(overrides=referee_overrides)

        # The goalkeeper (see `set_goalkeeper`): a permanently pinned slot,
        # entirely outside `_outfield_robot_ids`/`_slots` — no `Partitioner`
        # ever sees its robot, `_validate_partition` never has to account for
        # it, and `_barrier_reset` never touches it. `None` until
        # `set_goalkeeper` is called (goalkeeper-less callers, e.g. most of
        # `tests/engine/test_strategy.py`, are unaffected).
        self._pinned_slot: Optional[_TacticSlot] = None
        self._pinned_robot_id: Optional[RobotId] = None

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

    def set_goalkeeper(self, robot_id: int, tactic: Optional[Tactic] = None) -> None:
        """Pin `robot_id` to `tactic` (default `GoalkeeperTactic(robot_id)`) outside the scheduler entirely.

        Assigned post-construction the same way `match_log`/`referee_overrides`
        are — `AbstractStrategy` knows `goalkeeper_id` independently of
        whichever `build_kernel_strategy` factory built this `Strategy`, so
        this is the one place a caller needs to touch, not a constructor
        param threaded through every factory.

        The pinned slot is ticked unconditionally every tick (see `tick()`),
        with the same referee-gating a normal outfield tactic gets
        (frozen during HALT/STOP, exempted from most restart-formation
        overrides except kickoff — see `tick()`'s comments for exactly why),
        but is never handed to a `Partitioner`, never validated as part of
        the outfield partition's exhaustive cover, and never cleared by a
        barrier reset — "pinned" here means permanently, not just
        `is_committed()`-style revocably.
        """
        self._pinned_slot = _TacticSlot(tactic=tactic or GoalkeeperTactic(robot_id))
        self._pinned_slot.mem = self._pinned_slot.tactic.initial_mem()
        self._pinned_robot_id = robot_id

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
                override_commands = self._referee_override.tick(game, self._ctx.motion_controller, current_command)
                # Most override Steps still compute a command for every
                # friendly robot including the goalkeeper (e.g. `StopStep`
                # drives the keeper off the ball if it's inside the keep-out
                # radius; `PreparePenalty*Step` places it on the goal line) —
                # ticking the pinned tactic on top would overwrite that
                # intent with normal ball-tracking logic. `PrepareKickoff*Step`
                # is the exception: it exempts the goalkeeper from formation
                # entirely (a kickoff has no reason to pull it off its line),
                # so it's genuinely absent from `override_commands` during
                # those two commands specifically, and the pinned tactic
                # should tick normally to fill the gap — hence gating on
                # absence from the result rather than only on the command.
                if self._pinned_slot is not None and self._pinned_robot_id not in override_commands:
                    override_commands.update(self._tick_pinned(game))
                return override_commands

            if is_paused(current_command):
                # HALT: mem and commitments survive untouched, but no tactic
                # issues motion commands while play is stopped — including
                # the pinned slot, for the same reason.
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
                slot.commit_started_ts = None
                slot.commit_started_ball_pos = None

        commands: dict[RobotId, RobotCommand] = {}
        for tactic_id, robot_ids in partition.items():
            slot = self._slot_for(tactic_id)

            if slot.assigned_robots != robot_ids:
                slot.mem = slot.tactic.initial_mem() if robot_ids else None
                slot.assigned_robots = robot_ids
                slot.committed_ticks = 0
                slot.commit_started_ts = None
                slot.commit_started_ball_pos = None
                # A barrier reset (see `_barrier_reset`) wipes assigned_robots
                # unconditionally, including for slots the very next partition
                # just reassigns right back to their pre-reset robots — that
                # pairing is scheduler bookkeeping reasserting the status quo,
                # not a real reassignment, so it's not worth an intention-log
                # entry (the paired release from `_barrier_reset` was already
                # skipped for the same reason).
                reasserted = (
                    self._pending_barrier_reset is not None and self._pending_barrier_reset.get(tactic_id) == robot_ids
                )
                if self.match_log is not None and robot_ids and not reasserted:
                    self.match_log.intention(
                        tick=self._tick_count,
                        sim_time=getattr(game, "ts", 0.0),
                        tactic_id=tactic_id,
                        robot_ids=robot_ids,
                        tag=slot.tactic.tag,
                        note=type(slot.tactic).__name__,
                    )

            if not robot_ids:
                # A slot with no robots this tick has nothing to tick —
                # ticking a tactic with an empty robot set and no mem is
                # meaningless, not just a degenerate case of the normal path.
                continue

            ordered_robots = tuple(sorted(robot_ids))
            slot_commands, slot.mem = slot.tactic.tick(game, self._ctx, ordered_robots, slot.mem)
            commands.update(slot_commands)

            if self.match_log is not None:
                # Purely cosmetic per-tactic highlight (see `Tactic.highlights()`)
                # — logged under one shared key per slot so the dashboard/replay
                # canvas can render it without knowing every tactic id in advance.
                self.match_log.trace_if_changed(
                    tick=self._tick_count,
                    sim_time=getattr(game, "ts", 0.0),
                    key=f"highlights.{tactic_id}",
                    value=slot.tactic.highlights(slot.mem),
                )

        if self._pinned_slot is not None and self._pinned_robot_id not in commands:
            # Gated the same way the override branch gates it (see there):
            # in the ordinary case the pinned robot is never in
            # `_outfield_robot_ids` so this is always true, but a caller that
            # (unusually) also gives the pinned robot to an outfield tactic
            # must have that tactic's command win, not be silently
            # overwritten by the pinned tactic ticking on top of it.
            commands.update(self._tick_pinned(game))

        self._prev_partition = partition
        self._pending_barrier_reset = None
        return commands

    def _tick_pinned(self, game: Game) -> dict[RobotId, RobotCommand]:
        """Tick the pinned slot (see `set_goalkeeper`) and log its highlight trace.

        Never touches `_slots`/`_prev_partition`/`_pending_barrier_reset` —
        the pinned slot has no partition membership to reconcile, it simply
        runs every tick it's called.
        """
        slot = self._pinned_slot
        commands, slot.mem = slot.tactic.tick(game, self._ctx, (self._pinned_robot_id,), slot.mem)
        if self.match_log is not None:
            self.match_log.trace_if_changed(
                tick=self._tick_count,
                sim_time=getattr(game, "ts", 0.0),
                key="highlights.goalkeeper",
                value=slot.tactic.highlights(slot.mem),
            )
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

        **Commitment deadline (kernel-level stall breaker, not a play-length
        cap):** if `self._commitment_deadline_s` is set and a slot has been
        continuously committed for longer than it *and* the ball has not
        moved more than `_STALL_BALL_MOVEMENT_M` since the commitment began,
        this method treats the slot as NOT committed for this tick's
        partition decision — its robots go back into `free_robots` and the
        `Partitioner` is free to reassign them, exactly as if
        `is_committed()` had returned `False`. `Tactic.is_committed()` itself
        is never called differently and never told about this — the
        override happens only here, at the single point that decides what
        the `Partitioner` is told is free, so the single-writer partition
        invariant (design doc §5/§11) is untouched: the `Partitioner` is
        still the only thing deciding the partition, this only changes one
        input to it. If the ball HAS moved past the threshold, the deadline
        clock is irrelevant this tick — a commitment making real progress is
        never released just for running long.
        """
        pinned: dict[TacticId, frozenset[RobotId]] = {}
        for tactic_id, slot in self._slots.items():
            if not slot.assigned_robots:
                continue
            if slot.tactic.is_committed(game, slot.mem):
                if slot.committed_ticks == 0:
                    # Commitment just began this tick — anchor the deadline's
                    # reference point (elapsed time and ball position both
                    # measured from here, not from whenever the slot first
                    # got these robots).
                    slot.commit_started_ts = getattr(game, "ts", 0.0)
                    slot.commit_started_ball_pos = getattr(game, "ball", None)
                slot.committed_ticks += 1
                if slot.committed_ticks % 100 == 0:
                    logger.warning(
                        "tactic %r has blocked reassignment for %d consecutive ticks — "
                        "if this keeps climbing, its is_committed() logic is likely stuck",
                        tactic_id,
                        slot.committed_ticks,
                    )

                released = self._deadline_release(game, tactic_id, slot)
                if not released:
                    pinned[tactic_id] = slot.assigned_robots
            else:
                slot.committed_ticks = 0
                slot.commit_started_ts = None
                slot.commit_started_ball_pos = None

        pinned_robots = frozenset().union(*pinned.values()) if pinned else frozenset()
        free_robots = self._outfield_robot_ids - pinned_robots

        if self.match_log is not None:
            # Which robots are currently pinned by an is_committed() slot —
            # purely a display fact for the dashboard/replay canvas (see
            # design doc's single-writer invariant), not consulted by
            # anything in this class. Logged as a flat robot-id list, not
            # per-tactic-id, since a viewer wants "is this robot locked"
            # regardless of which slot holds it.
            self.match_log.trace_if_changed(
                tick=self._tick_count,
                sim_time=getattr(game, "ts", 0.0),
                key="committed_robot_ids",
                value=sorted(pinned_robots),
            )

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

    def _deadline_release(self, game: Game, tactic_id: TacticId, slot: _TacticSlot) -> bool:
        """True if `slot`'s commitment should be treated as released this tick
        under the commitment deadline (see `_choose_partition`'s docstring).

        On release, resets `slot.committed_ticks`/`commit_started_ts`/
        `commit_started_ball_pos`/`slot.mem` directly, right here —
        deliberately not left for `tick()`'s usual "robot set changed"
        reset. The `Partitioner` may well hand the exact same robots
        straight back this tick (nothing else wants them); `tick()` would
        then see `slot.assigned_robots == robot_ids` and skip its own reset
        entirely, which would otherwise leave two things wrong: the deadline
        clock exactly as expired as it was (re-releasing on every following
        tick forever — a tight loop, not the intended "one release, then a
        fresh deadline window"), and the tactic's `mem` exactly as it was
        when the stall was detected — the very state whose `is_committed()`
        never released on its own. Reassigning the identical robots to a
        tactic still holding the stuck mem that caused the stall in the
        first place would let it immediately re-declare committed off that
        same state, defeating the release. Resetting `mem` here mirrors what
        `tick()` already does for any other reassignment, just triggered by
        the deadline instead of a robot-set change.
        """
        if self._commitment_deadline_s is None:
            return False
        if slot.commit_started_ts is None:
            return False

        elapsed = getattr(game, "ts", 0.0) - slot.commit_started_ts
        if elapsed <= self._commitment_deadline_s:
            return False

        ball = getattr(game, "ball", None)
        ball_moved_m = 0.0
        if ball is not None and slot.commit_started_ball_pos is not None:
            try:
                ball_moved_m = ball.p.distance_to(slot.commit_started_ball_pos.p)
            except AttributeError:
                # A test/caller stub's `game.ball` doesn't expose the real
                # `.p`/`Vector*.distance_to` shape — treat as "can't tell the
                # ball moved," i.e. stalled, rather than silently never
                # releasing anything. Real `Game.ball` always has this shape.
                ball_moved_m = 0.0

        if ball_moved_m > _STALL_BALL_MOVEMENT_M:
            # Progress is being made — the deadline is a stall breaker, not a
            # play-length cap. Do not release; do not reset the clock either,
            # so a commitment that stalls again later is measured from its
            # original start, not restarted with a fresh grace window.
            return False

        if self.match_log is not None:
            self.match_log.intention(
                tick=self._tick_count,
                sim_time=getattr(game, "ts", 0.0),
                tactic_id=tactic_id,
                robot_ids=slot.assigned_robots,
                tag=slot.tactic.tag,
                note=(f"deadline release after {elapsed:.1f}s committed, ball moved {ball_moved_m:.3f}m"),
            )

        # Unconditional restart of the commitment clock and mem — see
        # docstring for why this can't wait for tick()'s own change detection.
        slot.committed_ticks = 0
        slot.commit_started_ts = None
        slot.commit_started_ball_pos = None
        slot.mem = slot.tactic.initial_mem()
        return True

    def _validate_partition(self, partition: dict[TacticId, frozenset[RobotId]]) -> None:
        """Enforce the single-writer invariant and reject picker bugs — but NOT
        an incomplete cover, which is a legitimate outcome, not a bug.

        A free robot can legitimately end up claimed by nobody: every
        registered Tactic can simultaneously be either committed elsewhere
        or `applicable() == False` for that robot's current situation (e.g.
        `PressAndContainTactic.applicable()` is range-gated and can go False
        for a lone free robot while every other slot is mid-commitment) —
        `Partitioner`s have no obligation to invent a tactic that wants an
        unwanted robot. `Strategy.tick()` simply doesn't tick a robot missing
        from every slot's `assigned_robots` that tick; the robot falls
        through to whatever the caller does for "no tactic addressed this
        robot" (`AbstractStrategy.execute_default_action`, a safe stop by
        default) — the exact same fallback path an unaddressed robot already
        takes for other reasons. Still a hard error, unchanged: a picker
        naming an unregistered tactic id, double-claiming one robot across
        two slots (breaks the single-writer invariant this whole class
        exists to guarantee), or inventing a robot id outside the outfield
        pool.
        """
        seen: set[RobotId] = set()
        for tactic_id, robots in partition.items():
            if tactic_id not in self._tactics:
                raise KeyError(f"picker chose unregistered tactic id {tactic_id!r}")
            overlap = seen & robots
            if overlap:
                raise ValueError(f"picker assigned robot(s) {sorted(overlap)} to more than one tactic in the same tick")
            seen |= robots

        extra = seen - self._outfield_robot_ids
        if extra:
            raise ValueError(f"picker assigned robot(s) {sorted(extra)} outside the outfield pool")

    def _barrier_reset(self, game: Game) -> None:
        pre_reset = {tid: slot.assigned_robots for tid, slot in self._slots.items() if slot.assigned_robots}
        for slot in self._slots.values():
            slot.mem = None
            slot.assigned_robots = frozenset()
            slot.committed_ticks = 0
            slot.commit_started_ts = None
            slot.commit_started_ball_pos = None
        self._prev_partition = None
        # Remembered so the very next tick's reassignment can tell "this is
        # the barrier's own mem/commitment wipe reasserting the same
        # partition" apart from a genuine reassignment, and skip logging the
        # reassignment half when it's a no-op restore. A referee command
        # flicker (e.g. STOP -> NORMAL_START -> STOP) otherwise produces two
        # intention-log entries every cycle even though nothing about which
        # robots are doing what actually changed.
        self._pending_barrier_reset = pre_reset if pre_reset else None
        if self.match_log is not None and pre_reset:
            self.match_log.intention(
                tick=self._tick_count,
                sim_time=getattr(game, "ts", 0.0),
                tactic_id="referee reset",
                robot_ids=(),
                tag=TacticTag.MIXED,
                note="referee restart cleared all tactic state",
            )
