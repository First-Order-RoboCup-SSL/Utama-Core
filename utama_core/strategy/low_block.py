"""The `low_block` kernel strategy: `build_low_block_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import fixed_ratio_picker
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.pass_and_shoot import PassAndShootTactic


def build_low_block_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory: a conservative, defense-heavy
    counterpart to `build_high_press_kernel_strategy` — most of the pool
    stays back regardless of possession (floored at `min_attack=2`, since
    `PassAndShootTactic` hard-requires at least 2 robots). Wires
    `PassAndShootTactic` ("attack") and `DefenseTactic` ("defense"), the
    original two ported Tactics, still useful as the minimal-risk baseline
    they were designed to be (§1/§2's original pairing) rather than the
    newer, more elaborate Tactics used elsewhere in this file.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": PassAndShootTactic(), "defense": DefenseTactic()},
            partitioner=fixed_ratio_picker("attack", "defense", attack_fraction=0.2, min_attack=2),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
