"""The `Tactic` contract — one tactic, written as a plain object with three methods.

No py_trees, no blackboard, no state-machine base class. `mem` is a plain
dataclass a Tactic defines for itself; the kernel never inspects its fields,
it only ever replaces it wholesale (via `make_initial_mem()`) or threads it
through unchanged (via `tick()`).

A `Tactic` does not know which robots it has until the kernel calls `tick()`
with them — robot assignment is the kernel's decision, not something a
Tactic can assume at construction time.
"""

from __future__ import annotations

from typing import Generic, Optional, Protocol, TypeVar

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext

MemT = TypeVar("MemT")

RobotId = int
TacticId = str


class Tactic(Protocol[MemT]):
    """Structural contract a tactic must satisfy. Not required to subclass this."""

    def make_initial_mem(self) -> MemT:
        """Fresh state for this tactic. Called whenever its robot assignment changes."""
        ...

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: MemT
    ) -> tuple[dict[RobotId, RobotCommand], MemT]:
        """Compute this tick's commands for `robot_ids` and the next `mem`."""
        ...

    def committed(self, game: Game, mem: MemT) -> bool:
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
    """Convenience base providing the two defaults `Tactic` doesn't strictly need.

    Purely optional — a tactic only needs to structurally match `Tactic`, not
    inherit from anything. This exists so most tactics can skip writing
    `committed`/`suggest_next` boilerplate when the defaults are fine.
    """

    def committed(self, game: Game, mem: MemT) -> bool:
        return False

    def suggest_next(self, game: Game, mem: MemT) -> Optional[TacticId]:
        return None
