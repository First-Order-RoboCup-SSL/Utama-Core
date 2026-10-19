"""The OpenJev flat engine's scripted rule as a catalog strategy (`docs/openjev.md`).

One tactic owns all five outfield robots and, at 1 Hz and at once when play changes, gives each an
action from its own menu, the carrier first: shoot / pass / dribble / clear with the ball; go for the
ball, move to a cell, mark, block the shot or hold without it (`tactics/flat.py`'s `scripted_choice`).
The engine's own free-kick formations take its restarts. It is what that engine plays when no model
answers, so it needs no server.
"""

from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.flat import FlatJevTactic, flat_strategy


def build_flat_rule_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """The flat engine on its scripted rule. Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the same name."""

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        return flat_strategy(outfield_robot_ids, FlatJevTactic(None, policy="scripted"), motion_controller)

    return _build
