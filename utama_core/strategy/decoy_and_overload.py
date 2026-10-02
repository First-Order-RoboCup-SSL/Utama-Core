"""The `decoy_and_overload` kernel strategy: `build_decoy_and_overload_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import fixed_ratio_picker
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def build_decoy_and_overload_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory exercising `DecoyOverloadTactic`.

    Wires `DecoyOverloadTactic` ("attack") and `ShadowAndMarkTactic`
    ("defense"), split by `fixed_ratio_picker` with `min_attack=2` —
    `DecoyOverloadTactic` hard-requires at least 2 robots (a decoy and an
    overloader) the same way `PassAndShootTactic` does, so it needs the
    same floor `build_low_block_kernel_strategy` gives that tactic. Attack
    fraction left at a plain 0.5 split (possession-agnostic) since nothing
    about the lure/overload pattern is more or less urgent when the
    opponent has the ball, unlike the press/shadow pairings above.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": DecoyOverloadTactic(), "defense": ShadowAndMarkTactic()},
            partitioner=fixed_ratio_picker("attack", "defense", attack_fraction=0.5, min_attack=2),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
