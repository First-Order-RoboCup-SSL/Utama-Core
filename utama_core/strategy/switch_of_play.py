"""The `switch_of_play` kernel strategy: `build_switch_of_play_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import _fixed_ratio_picker
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.switch_of_play import SwitchOfPlayTactic


def build_switch_of_play_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory exercising `SwitchOfPlayTactic`.

    Wires `SwitchOfPlayTactic` ("attack") and `DefenseTactic` ("defense"),
    split by `_fixed_ratio_picker` with `min_attack=3` — the tactic's full
    carrier/pivot/runner relay needs 3 robots to actually exercise the
    "switch" leg (the pivot outlet) rather than silently degrading to its
    2-robot direct-pass fallback every tick, the same floor rationale
    `build_low_block_kernel_strategy` gives `PassAndShootTactic` and
    `build_decoy_and_overload_kernel_strategy` gives `DecoyOverloadTactic`.
    Attack fraction left at a plain 0.5 split (possession-agnostic), same as
    `build_decoy_and_overload_kernel_strategy` — nothing about reading the
    weak side and relaying the ball across it is more or less urgent when the
    opponent has the ball.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": SwitchOfPlayTactic(), "defense": DefenseTactic()},
            partitioner=_fixed_ratio_picker("attack", "defense", attack_fraction=0.5, min_attack=3),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
