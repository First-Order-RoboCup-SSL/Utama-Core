"""The `give_and_go_solo` kernel strategy: `build_give_and_go_solo_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.give_and_go import GiveAndGoTactic


def build_give_and_go_solo_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Single-tactic `Strategy` factory: the entire outfield pool always runs
    `GiveAndGoTactic`, no defense slot at all. Mirrors
    `build_default_kernel_strategy`'s shape (single Tactic, no real
    allocation decision, via `single_tactic_picker`) but for the newer
    passing tactic — useful as an isolated benchmark/eval config when
    testing `GiveAndGoTactic` in isolation matters more than realistic match
    posture (e.g. tuning `_MAX_HOPS_PER_POSSESSION` or the support-scoring
    weights without a defensive Tactic's behaviour as a confound).

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": GiveAndGoTactic()},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "attack"),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
