"""Everything a tactic's `tick()` needs besides `Game`, its assigned robots, and `mem`.

Kept as an explicit, typed object instead of blackboard lookups so a
tactic's inputs are visible in its function signature. Ported unchanged
from `utama_strategy.functional.core.TickContext` — briefly renamed
`KernelContext` during the kernel-tactic rewrite, restored to its original
name 2026-08-26 (see `docs/roadmap.md` item 3): "kernel" wasn't
disambiguating anything (most of this module namespace is already
kernel-something), while `TickContext` says exactly what this is — the
per-tick auxiliary inputs a `Tactic.tick()` call needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from utama_core.motion_planning.src.common.motion_controller import MotionController

if TYPE_CHECKING:
    from utama_core.engine.match_log import MatchLog


@dataclass
class TickContext:
    motion_controller: MotionController
    # Set by `Strategy` from its own `match_log` (see `Strategy.match_log`'s
    # setter) so a tactic/skill can call `ctx.match_log.trace(...)` without a
    # separate logger threaded through every call site. `None` whenever
    # match logging is disabled (the common case — tournament/CI runs) so
    # tracing a tick is a single `is not None` check, not a real cost.
    match_log: Optional["MatchLog"] = None
