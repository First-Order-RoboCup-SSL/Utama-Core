"""The `counter_press` kernel strategy: `build_counter_press_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import (
    _allocate_ordered,
    _carrier_first,
    _friendly_closer_to_ball,
)
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.switch_of_play import SwitchOfPlayTactic


def _counter_press_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Counter-press posture: all-out pressure when we lose it, low block when
    there is nothing to press, four-up on the switch when we regain it.

    - Opponent has the ball and pressing is possible: everyone presses
      (1 ball-presser + everyone else man-marking) — the ball is smothered
      where it was lost.
    - Opponent has the ball but nothing is pressable: everyone holds the
      `block_shape` zone screen — the compact low block.
    - We have the ball: the switch-of-play needs its three roles
      (carrier/pivot/runner), so 4 robots attack through the weak side while
      1 keeps the screen shape as insurance on the counter.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True

    press_ok = "press" in available_tactic_ids
    attack_ok = "attack" in available_tactic_ids
    block_ok = "block" in available_tactic_ids

    if losing:
        if press_ok:
            return _allocate_ordered(ordered, "press", len(ordered))
        if block_ok:
            return _allocate_ordered(ordered, "block", len(ordered))
        if attack_ok:
            return _allocate_ordered(ordered, "attack", len(ordered))
        return {}
    if attack_ok:
        return _allocate_ordered(ordered, "attack", 4, "block" if block_ok else None)
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def build_counter_press_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """High-intensity transition team: smother the ball where it was lost,
    then switch the play to the weak side at full commitment.

    Three concurrent slots — `SwitchOfPlayTactic` ("attack"),
    `PressAndContainTactic` ("press"), and the new `BlockShapeTactic`
    ("block") — allocated by `_counter_press_picker`: everyone presses while
    the opponent has it (and something is pressable), everyone drops into
    the zone screen when they are shielded from the press, and 4 robots
    attack through the pivot/runner switch the moment the ball is won. The
    three-role switch (carrier/pivot/runner) needs at least 3 robots, hence
    the 4+1 split.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": SwitchOfPlayTactic(),
                "press": PressAndContainTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_counter_press_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
