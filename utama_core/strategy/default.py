"""The `default` kernel strategy: `default` and the pickers only it uses."""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.pass_and_shoot import PassAndShootTactic


def build(outfield_robot_ids: tuple[int, ...], motion_controller: MotionController) -> KernelSchedulerStrategy:
    """Minimal, single-tactic-pool `Strategy` factory: everyone attacks.

    Only one real multi-robot tactic exists so far (`pass_and_shoot`), so
    the picker has nothing to choose between yet — per the design doc's
    "splitting policy" deferral, this deliberately does not invent an
    allocation policy ahead of a second concrete tactic that would need one.
    Callers with more than one outfield tactic should build their own
    `Strategy` with a real `Picker` instead of using this helper.

    Returns a `build(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    ctx = TickContext(motion_controller=motion_controller)
    return KernelSchedulerStrategy(
        tactics={"pass_and_shoot": PassAndShootTactic()},
        partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "pass_and_shoot"),
        outfield_robot_ids=outfield_robot_ids,
        ctx=ctx,
    )
