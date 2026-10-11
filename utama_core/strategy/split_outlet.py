"""The `split_outlet` kernel strategy: `split_shape` plus a counter-attack outlet slot.

`split_shape` gives its defence one robot whenever it holds the ball edge; `split_outlet`
keeps that split unchanged. Without the ball edge (and with 3 or more robots) it takes
one robot from the defence for an `OutletTactic` slot, which holds on the weak side of
the attacking half so a turnover has a receiver ahead of the defensive line. The outlet
robot rejoins the attack whenever the ball edge is ours.
"""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import carrier_first
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.outlet import OutletTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def _split_outlet_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`split_shape`'s split, with an outlet slot taken from the defence when we lack the ball edge.

    Counts (n = robots): with the ball edge, or with fewer than 3 robots, the split is
    `split_shape`'s exactly and the outlet slot is never used. Otherwise attack 1,
    outlet 1, defence n - 2. A slot missing from `available_tactic_ids` hands its
    robots to defence, then attack, then outlet.

    Only emits a key for a slot it assigns robots to, never a zero-robot entry (see
    `split_shape`'s picker for why).
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}
    n = len(ordered)

    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    friendly_has_ball_edge = friendly_dist < enemy_dist

    if friendly_has_ball_edge or n < 3:
        attack_count = (n + 1) // 2 + 1 if friendly_has_ball_edge else n // 2 - 1
        attack_count = max(0, min(n, attack_count))
        counts = {"attack": attack_count, "outlet": 0, "defense": n - attack_count}
    else:
        counts = {"attack": 1, "outlet": 1, "defense": n - 2}

    available = {slot: count for slot, count in counts.items() if slot in available_tactic_ids and count > 0}
    spill = sum(count for slot, count in counts.items() if slot not in available_tactic_ids)
    if spill:
        target = next((slot for slot in ("defense", "attack", "outlet") if slot in available), None)
        if target is None:
            return {}
        available[target] = available.get(target, 0) + spill

    partition: dict[str, frozenset[RobotId]] = {}
    start = 0
    for slot in ("attack", "outlet", "defense"):
        count = available.get(slot, 0)
        if count > 0:
            partition[slot] = frozenset(ordered[start : start + count])
            start += count
    return partition


def build_split_outlet_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) split-outlet `Strategy` factory.

    Wires `LeadAndSupportTactic` ("attack"), `OutletTactic` ("outlet") and
    `ShadowAndMarkTactic` ("defense") as concurrently active slots, split by
    `_split_outlet_picker`.

    Returns a `build_kernel_strategy(motion_controller)` callable suitable for
    `AbstractStrategy`'s constructor argument of the same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": LeadAndSupportTactic(),
                "outlet": OutletTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_split_outlet_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
