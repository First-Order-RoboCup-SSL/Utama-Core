"""The `switch_of_play` kernel strategy: `switch_of_play` and the pickers only it uses."""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import fixed_ratio_picker
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.switch_of_play import SwitchOfPlayTactic


def build(outfield_robot_ids: tuple[int, ...], motion_controller: MotionController) -> KernelSchedulerStrategy:
    """5-robot (or fewer) `Strategy` factory exercising `SwitchOfPlayTactic`.

    Wires `SwitchOfPlayTactic` ("attack") and `DefenseTactic` ("defense"),
    split by `fixed_ratio_picker` with `min_attack=3` — the tactic's full
    carrier/pivot/runner relay needs 3 robots to actually exercise the
    "switch" leg (the pivot outlet) rather than silently degrading to its
    2-robot direct-pass fallback every tick, the same floor rationale
    `low_block` gives `PassAndShootTactic` and
    `decoy_and_overload` gives `DecoyOverloadTactic`.
    Attack fraction left at a plain 0.5 split (possession-agnostic), same as
    `decoy_and_overload` — nothing about reading the
    weak side and relaying the ball across it is more or less urgent when the
    opponent has the ball.

    Returns a `build(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    ctx = TickContext(motion_controller=motion_controller)
    return KernelSchedulerStrategy(
        tactics={"attack": SwitchOfPlayTactic(), "defense": DefenseTactic()},
        partitioner=fixed_ratio_picker("attack", "defense", attack_fraction=0.5, min_attack=3),
        outfield_robot_ids=outfield_robot_ids,
        ctx=ctx,
    )
