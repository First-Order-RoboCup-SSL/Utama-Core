"""The `Tactic` contract — one tactic, written as a plain object with a few methods.

No py_trees, no blackboard, no state-machine base class. `mem` is a plain
dataclass a Tactic defines for itself; the kernel never inspects its fields,
it only ever replaces it wholesale (via `initial_mem()`) or threads it
through unchanged (via `tick()`).

A `Tactic` does not know which robots it has until the kernel calls `tick()`
with them — robot assignment is the kernel's decision, not something a
Tactic can assume at construction time.
"""

from __future__ import annotations

import enum
from typing import Generic, Optional, Protocol, TypeVar

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext

MemT = TypeVar("MemT")

RobotId = int
TacticId = str


class TacticTag(enum.Enum):
    """A closed set of coarse roles a `Tactic` plays, for `Partitioner`s to allocate by.

    Deliberately closed, not an open/extensible vocabulary — see design doc
    §15. A `Partitioner` can allocate robots by tag (e.g. "give ATTACK-tagged
    candidates 60% of the free pool") instead of hardcoded Tactic-id string
    branches, so a new same-tagged Tactic doesn't require touching any
    Partitioner. Revisit only once a concrete Tactic genuinely doesn't fit
    any of these three, not in anticipation of one.
    """

    ATTACK = "attack"
    DEFENSE = "defense"
    MIXED = "mixed"


class Tactic(Protocol[MemT]):
    """Structural contract a tactic must satisfy. Not required to subclass this."""

    tag: TacticTag
    """Which of the closed set of roles this tactic plays — see `TacticTag`."""

    def initial_mem(self) -> MemT:
        """Fresh state for this tactic. Called whenever its robot assignment changes."""
        ...

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: MemT
    ) -> tuple[dict[RobotId, RobotCommand], MemT]:
        """Compute this tick's commands for `robot_ids` and the next `mem`."""
        ...

    def applicable(self, game: Game) -> bool:
        """True if it is sensible for this tactic to *begin* running right now.

        Self-declared, like `is_committed()`, but answers a different
        question: a precondition checked on entry to a slot, not a
        protection against being interrupted mid-action. Queried fresh every
        tick for every currently-unassigned, non-`is_committed()` tactic,
        before the free robot pool ever reaches a `Partitioner` — a `False`
        here removes the tactic from that tick's candidates entirely, so a
        `Partitioner` never has to know how to avoid a tactic that doesn't
        make sense right now.

        Never consulted for a slot that is currently `is_committed()`: a
        commitment protects an in-progress action regardless of whether the
        tactic would still consider itself applicable if asked — see design
        doc §15. The tick after a commitment ends, `applicable()` is asked
        for the first time and evicts the tactic then if it says False; no
        special-casing is needed for "committed but no longer applicable."

        Default: always applicable, so existing tactics need no changes to
        keep working.
        """
        return True

    def is_committed(self, game: Game, mem: MemT) -> bool:
        """True if the kernel must not reassign this tactic's robots right now.

        Self-declared and absolute: while this returns True, the scheduler
        will not move this tactic's robots to another tactic, with one
        exception — a barrier reset (see `kernel.referee_reset`) clears every
        tactic's commitment unconditionally, because a referee restart makes
        the previous tick's in-progress action moot for everyone at once, not
        because the scheduler forced this tactic off its robots.

        Default: never committed, freely reassignable at any time.
        """
        return False

    def suggest_next(self, game: Game, mem: MemT) -> Optional[TacticId]:
        """Optional, purely advisory hint for what the scheduler might run next.

        The scheduler is free to ignore this. There is no priority/scoring
        system behind it — it exists only so a tactic that knows it is about
        to finish can say so, without being able to force anything.

        Default: no opinion.
        """
        return None


class BaseTactic(Generic[MemT]):
    """Convenience base providing the defaults `Tactic` doesn't strictly need.

    Purely optional — a tactic only needs to structurally match `Tactic`, not
    inherit from anything. This exists so most tactics can skip writing
    `applicable`/`is_committed`/`suggest_next` boilerplate when the defaults
    are fine. Does NOT default `tag` — every tactic must declare its own.
    """

    def applicable(self, game: Game) -> bool:
        return True

    def is_committed(self, game: Game, mem: MemT) -> bool:
        return False

    def suggest_next(self, game: Game, mem: MemT) -> Optional[TacticId]:
        return None
