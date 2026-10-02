"""The `three_slot` kernel strategy: `build_three_slot_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import _carrier_first
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def _three_way_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Splits the free pool three ways — first exercise of `Strategy` running
    more than two concurrent slots (nothing in `Strategy`/`_validate_partition`
    is hardcoded to two, but no config before this one had actually tried
    three). One presser, one shadow-and-mark defensive pair when there are
    enough robots to spare, everyone else attacks via give-and-go. Falls back
    to redistributing a slot's share to "attack" whenever that slot's Tactic
    is currently inapplicable or pinned elsewhere, same defensive pattern as
    `_press_and_pass_split_picker`.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    press_ok = "press" in available_tactic_ids
    mark_ok = "mark" in available_tactic_ids
    attack_ok = "attack" in available_tactic_ids

    remaining = list(ordered)
    partition: dict[str, frozenset[RobotId]] = {}

    if press_ok and remaining:
        partition["press"] = frozenset([remaining.pop(0)])
    if mark_ok and len(remaining) >= 2:
        partition["mark"] = frozenset(remaining[:2])
        remaining = remaining[2:]
    if attack_ok:
        if remaining:
            partition["attack"] = frozenset(remaining)
    elif remaining:
        # "attack" unavailable too — nothing left to hand the rest to; leave
        # them off the partition only if every other slot already claimed
        # everyone, otherwise this would violate the exhaustive-cover
        # invariant, so fall back to whichever defensive slot is still open.
        if mark_ok:
            partition["mark"] = frozenset(set(partition.get("mark", frozenset())) | set(remaining))
        elif press_ok:
            partition["press"] = frozenset(set(partition.get("press", frozenset())) | set(remaining))

    return partition


def build_three_slot_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory running three concurrent Tactic
    slots at once: `PressAndContainTactic` ("press"), `ShadowAndMarkTactic`
    ("mark"), and `GiveAndGoTactic` ("attack"). Distinct in kind from every
    other factory in this file, which all run exactly two slots — this is
    the first config to actually exercise `Strategy` with N>2, which the
    kernel has always structurally supported (`_validate_partition` iterates
    `partition.items()` generically) but nothing had tested until now.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "press": PressAndContainTactic(),
                "mark": ShadowAndMarkTactic(),
                "attack": GiveAndGoTactic(),
            },
            partitioner=_three_way_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
