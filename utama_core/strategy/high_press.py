"""The `high_press` kernel strategy: `build_high_press_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import _fixed_ratio_picker
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic


def build_high_press_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory: an aggressive, *non-reactive*
    posture — most of the pool commits forward regardless of who has the
    ball, unlike `build_press_and_pass_kernel_strategy`'s possession-edge
    split. Wires the same `GiveAndGoTactic`/`PressAndContainTactic` pair as
    that config, but the two are not interchangeable: the point here is to
    demonstrate that the same Tactic set can be driven by a mechanically
    different `Partitioner` (fixed ratio, via `_fixed_ratio_picker`) — the
    scheduling *policy* is what varies between example strategies, not just
    the Tactic roster.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": GiveAndGoTactic(), "defense": PressAndContainTactic()},
            partitioner=_fixed_ratio_picker("attack", "defense", attack_fraction=0.8),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
