"""Everything a tactic's `tick()` needs besides `Game`, its assigned robots, and `mem`.

Kept as an explicit, typed object instead of blackboard lookups so a
tactic's inputs are visible in its function signature. Ported unchanged
(renamed) from `utama_strategy.functional.core.TickContext`.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.motion_planning.src.common.motion_controller import MotionController


@dataclass
class KernelContext:
    motion_controller: MotionController
    rsim_env: object | None = None
