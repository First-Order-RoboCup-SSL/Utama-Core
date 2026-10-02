"""The `zone_fluid` kernel strategy: `build_zone_fluid_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import (
    _allocate_ordered,
    _ball_zone,
    _carrier_first,
    _friendly_closer_to_ball,
)
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def _zone_flow_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Zone-flow posture: the attacking *pattern* changes with ball zone, the
    defense stays man-shaped throughout.

    - Opponent has the ball: everyone shadows/marks (the whole team in
      shape).
    - We have the ball in our own or middle third: a give-and-go trio works
      the ball forward while 2 shadow/mark.
    - We have the ball in the final third: the decoy-and-overload pair
      lures the last line out of position (2 robots — that tactic is a
      two-role duet by design) while the other 3 hold the defensive shape.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True

    defense_ok = "defense" in available_tactic_ids
    givego_ok = "givego" in available_tactic_ids
    overload_ok = "overload" in available_tactic_ids

    if losing:
        if defense_ok:
            return _allocate_ordered(ordered, "defense", len(ordered))
        if givego_ok:
            return _allocate_ordered(ordered, "givego", len(ordered))
        if overload_ok:
            return _allocate_ordered(ordered, "overload", len(ordered))
        return {}

    zone = _ball_zone(game)
    if zone == "final" and overload_ok:
        return _allocate_ordered(ordered, "overload", 2, "defense" if defense_ok else None)
    if givego_ok:
        return _allocate_ordered(ordered, "givego", 3, "defense" if defense_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if defense_ok:
        return _allocate_ordered(ordered, "defense", len(ordered))
    return {}


def build_zone_fluid_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Zone-adaptive team: the attacking pattern itself changes with ball
    position, the first strategy in the catalog with two concurrent ATTACK-
    tagged slots that the picker chooses between — exactly the use the closed
    `TacticTag` vocabulary exists for.

    Three concurrent slots — `GiveAndGoTactic` ("givego"),
    `DecoyOverloadTactic` ("overload"), and `ShadowAndMarkTactic`
    ("defense") — allocated by `_zone_flow_picker`: the whole team takes
    man-shape when the ball is lost; the give-and-go trio builds up through
    the middle thirds; and in the final third the two-robot decoy/overload
    duet replaces the trio, luring the last line out of position instead of
    passing into it.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "givego": GiveAndGoTactic(),
                "overload": DecoyOverloadTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_zone_flow_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
