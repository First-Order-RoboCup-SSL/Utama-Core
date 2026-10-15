"""The `split_sticky` kernel strategy: `build_split_sticky_kernel_strategy` and the picker only it uses.

Same two slots and split sizes as `split_shape` (`LeadAndSupportTactic` "attack",
`ShadowAndMarkTactic` "defense"). The one change is a sticky possession edge:
`split_shape` re-reads which side is closer to the ball every tick, so its
posture flickers in contested balls. Here, holding the attacking posture
survives a near-tie and is only given up on a clear loss; switching into it
needs a clear gain.
"""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import carrier_first, friendly_closer_to_ball
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def _sticky_split_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Whichever side is closer to the ball decides posture, with hysteresis on the edge.

    "Holding attack" means last tick's attack slot had more robots than defense.
    While holding it, only a clear loss (`friendly_closer_to_ball` is False)
    gives it up; a near-tie (None) keeps it. Not holding it, only a clear gain
    (True) takes it. Split sizes are `split_shape`'s: attack (n+1)//2+1 when
    attacking, n//2-1 otherwise.

    A slot missing from `available_tactic_ids` is pinned by a commitment, and
    its share goes to the other slot. Only emits a key for a tactic it is
    actually assigning robots to, never a zero-robot entry.
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    holding_attack = len(prev_partition.get("attack", ())) > len(prev_partition.get("defense", ()))
    edge = friendly_closer_to_ball(game)
    attacking = edge is not False if holding_attack else edge is True

    if "attack" not in available_tactic_ids:
        return {"defense": frozenset(ordered)} if "defense" in available_tactic_ids else {}
    if "defense" not in available_tactic_ids:
        return {"attack": frozenset(ordered)}

    attack_count = (len(ordered) + 1) // 2 + 1 if attacking else len(ordered) // 2 - 1
    attack_count = max(0, min(len(ordered), attack_count))

    partition = {}
    if attack_count > 0:
        partition["attack"] = frozenset(ordered[:attack_count])
    if attack_count < len(ordered):
        partition["defense"] = frozenset(ordered[attack_count:])
    return partition


def build_split_sticky_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) sticky split-shape `Strategy` factory.

    Wires `LeadAndSupportTactic` ("attack") and `ShadowAndMarkTactic`
    ("defense") as two concurrently active slots, split by `_sticky_split_picker`.

    Returns a `build_kernel_strategy(motion_controller)` callable suitable for
    `AbstractStrategy`'s constructor argument of the same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": LeadAndSupportTactic(), "defense": ShadowAndMarkTactic()},
            partitioner=_sticky_split_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
