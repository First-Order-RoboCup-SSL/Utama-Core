"""The `zone_split_shape` kernel strategy: `split_shape`'s two tactics with a split that reads the ball's third."""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import (
    ball_zone,
    carrier_first,
    friendly_closer_to_ball,
)
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def _attack_count(n: int, edge: Optional[bool], zone: str, prev_attack: Optional[int]) -> int:
    """How many of `n` free robots attack.

    With the ball: 4 of 5 in the middle and final thirds, but one fewer in our own third, so a
    build-out from the back keeps cover. Without it: 1 of 5 when the ball is in our own or middle
    third, but 2 when it is in their final third, so a robot stays up to win the ball back there.
    A near-tie (`edge` None) keeps the previous count.
    """
    if edge is None:
        if prev_attack is not None:
            return max(0, min(n, prev_attack))
        edge = False
    if edge:
        count = (n + 1) // 2 + 1
        if zone == "own":
            count -= 1
    else:
        count = n // 2 - 1
        if zone == "final":
            count += 1
    return max(0, min(n, count))


def _zone_split_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`split_shape`'s possession split, shifted one robot by where the ball is. See `_attack_count`."""
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}
    if "attack" not in available_tactic_ids:
        return {"defense": frozenset(ordered)} if "defense" in available_tactic_ids else {}
    if "defense" not in available_tactic_ids:
        return {"attack": frozenset(ordered)}

    prev_attack = len(prev_partition["attack"]) if prev_partition and "attack" in prev_partition else None
    attack_count = _attack_count(len(ordered), friendly_closer_to_ball(game), ball_zone(game), prev_attack)

    partition = {}
    if attack_count > 0:
        partition["attack"] = frozenset(ordered[:attack_count])
    if attack_count < len(ordered):
        partition["defense"] = frozenset(ordered[attack_count:])
    return partition


def build_zone_split_shape_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """`split_shape` with the split shifted by the ball's third of the pitch."""

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": LeadAndSupportTactic(), "defense": ShadowAndMarkTactic()},
            partitioner=_zone_split_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
