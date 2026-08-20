"""Everything a tactic's `tick()` needs besides `Game`, its assigned robots, and `mem`.

Kept as an explicit, typed object instead of blackboard lookups so a
tactic's inputs are visible in its function signature. Ported unchanged
(renamed) from `utama_strategy.functional.core.TickContext`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from utama_core.motion_planning.src.common.motion_controller import MotionController

if TYPE_CHECKING:
    from utama_core.kernel.match_log import MatchLog


@dataclass
class KernelContext:
    motion_controller: MotionController
    # Set by `Strategy` from its own `match_log` (see `Strategy.match_log`'s
    # setter) so a tactic/skill can call `ctx.match_log.trace(...)` without a
    # separate logger threaded through every call site. `None` whenever
    # match logging is disabled (the common case — tournament/CI runs) so
    # tracing a tick is a single `is not None` check, not a real cost.
    match_log: Optional["MatchLog"] = None
