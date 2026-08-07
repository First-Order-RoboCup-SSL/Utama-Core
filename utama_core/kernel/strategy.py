"""`Strategy` — the per-team-color scheduler that decides which `Tactic` runs.

Single active tactic at a time, claiming the whole outfield robot pool
(robot 0 / goalkeeper is pinned separately and never scheduled). No
concurrent multi-tactic partitioning, no bid/fitness scoring, no state
machine — a plain, hand-written picker function decides which tactic is
active each tick, and this class enforces the invariants around that
decision: `committed()` is an absolute veto, `mem` resets exactly when a
tactic's assigned robot set changes (compared as sets, not tuples), and a
referee barrier reset clears everything unconditionally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.referee_reset import ResetTier, classify_transition, is_paused
from utama_core.kernel.tactic import RobotId, Tactic, TacticId

logger = logging.getLogger(__name__)

# A picker decides which tactic should be active this tick, given the game
# state and the currently active tactic (None on the very first tick or
# right after a reset). It returns the TacticId to activate. It does NOT
# decide the robot pool — that's a fixed, kernel-owned set handed to
# `Strategy` at construction (everything but the pinned goalkeeper).
Picker = Callable[[Game, Optional[TacticId]], TacticId]


@dataclass
class _TacticSlot:
    tactic: Tactic
    mem: object = None
    assigned_robots: frozenset[RobotId] = field(default_factory=frozenset)
    committed_ticks: int = 0


class Strategy:
    def __init__(
        self,
        tactics: dict[TacticId, Tactic],
        picker: Picker,
        outfield_robot_ids: tuple[RobotId, ...],
        ctx: KernelContext,
    ):
        if not tactics:
            raise ValueError("Strategy needs at least one registered tactic")
        self._slots: dict[TacticId, _TacticSlot] = {tid: _TacticSlot(tactic=t) for tid, t in tactics.items()}
        self._picker = picker
        self._outfield_robot_ids = tuple(outfield_robot_ids)
        self._ctx = ctx

        self._active_tactic_id: Optional[TacticId] = None
        self._prev_referee_command = None

    @property
    def active_tactic_id(self) -> Optional[TacticId]:
        return self._active_tactic_id

    def tick(self, game: Game) -> dict[RobotId, RobotCommand]:
        referee = getattr(game, "referee", None)
        current_command = getattr(referee, "referee_command", None) if referee is not None else None

        if current_command is not None:
            tier = classify_transition(self._prev_referee_command, current_command)
            if tier is ResetTier.BARRIER:
                self._barrier_reset()
            self._prev_referee_command = current_command

            if is_paused(current_command):
                # Pause: mem and commitments survive untouched, but no tactic
                # issues motion commands while play is stopped.
                return {}

        active_id = self._choose_active_tactic(game)
        slot = self._slots[active_id]

        if slot.assigned_robots != frozenset(self._outfield_robot_ids):
            slot.mem = slot.tactic.make_initial_mem()
            slot.assigned_robots = frozenset(self._outfield_robot_ids)
            slot.committed_ticks = 0

        commands, slot.mem = slot.tactic.tick(game, self._ctx, self._outfield_robot_ids, slot.mem)
        self._active_tactic_id = active_id
        return commands

    def _choose_active_tactic(self, game: Game) -> TacticId:
        if self._active_tactic_id is not None:
            active_slot = self._slots[self._active_tactic_id]
            if active_slot.tactic.committed(game, active_slot.mem):
                active_slot.committed_ticks += 1
                if active_slot.committed_ticks % 100 == 0:
                    logger.warning(
                        "tactic %r has blocked reassignment for %d consecutive ticks — "
                        "if this keeps climbing, its committed() logic is likely stuck",
                        self._active_tactic_id,
                        active_slot.committed_ticks,
                    )
                return self._active_tactic_id
            active_slot.committed_ticks = 0

        candidate_id = self._picker(game, self._active_tactic_id)
        if candidate_id not in self._slots:
            raise KeyError(f"picker chose unregistered tactic id {candidate_id!r}")
        return candidate_id

    def _barrier_reset(self) -> None:
        for slot in self._slots.values():
            slot.mem = None
            slot.assigned_robots = frozenset()
            slot.committed_ticks = 0
        self._active_tactic_id = None
