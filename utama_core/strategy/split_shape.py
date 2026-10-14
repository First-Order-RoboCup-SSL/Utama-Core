"""The `split_shape` kernel strategy: `build_split_shape_kernel_strategy` and the pickers only it uses."""

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
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def _possession_split_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Whichever side is closer to the ball decides posture; the split is fixed once decided.

    Neither `LeadAndSupportTactic` nor `ShadowAndMarkTactic` overrides
    `applicable()`, so a slot is missing from `available_tactic_ids` only
    while it's pinned by a commitment — its share then goes to the other slot.

    Deliberately the simplest rule that gives the split-shape scheduler
    something real to react to, not a scored/tunable allocator (see the
    design doc's stance against bid/fitness-scoring machinery) — one signal
    (which team is closer to the ball), two fixed splits. Ties and an
    unreadable proximity lookup default to the more conservative
    (defense-heavy) split.

    Only emits a key for a tactic it is actually assigning free robots to —
    never a zero-robot entry, since `Strategy` treats every key in a
    `Partitioner`'s return value as "I am claiming this tactic id right now,"
    and a present-but-empty entry for a tactic committed and pinned
    elsewhere would collide with that pin.
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}

    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    friendly_has_ball_edge = friendly_dist < enemy_dist

    if "attack" not in available_tactic_ids:
        return {"defense": frozenset(ordered)} if "defense" in available_tactic_ids else {}
    if "defense" not in available_tactic_ids:
        return {"attack": frozenset(ordered)}

    attack_count = (len(ordered) + 1) // 2 + 1 if friendly_has_ball_edge else len(ordered) // 2 - 1
    attack_count = max(0, min(len(ordered), attack_count))

    partition = {}
    if attack_count > 0:
        partition["attack"] = frozenset(ordered[:attack_count])
    if attack_count < len(ordered):
        partition["defense"] = frozenset(ordered[attack_count:])
    return partition


def build_split_shape_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) split-shape `Strategy` factory.

    Wires `LeadAndSupportTactic` ("attack") and `ShadowAndMarkTactic`
    ("defense") as two concurrently active slots, split by
    `_possession_split_picker`. This is the concrete forcing case the design
    doc's §7 deferral was waiting on — see §11 for the full rationale.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": LeadAndSupportTactic(), "defense": ShadowAndMarkTactic()},
            partitioner=_possession_split_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
