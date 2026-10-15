"""The `split_press` kernel strategy: `build_split_press_kernel_strategy` and the pickers only it uses.

Differs from `split_shape` in one place: without the ball edge, the lone chaser
(the leader of `LeadAndSupportTactic`, which just walked to the ball) is replaced
by a 2-robot `PressAndContainTactic` slot ("press") while an enemy is within
press range. Split_shape's robots got close to the ball but did not win it, so
the press pairs a presser with a marker on the likely receiver. With the ball
edge, the split is split_shape's unchanged.
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
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic

_PRESS_ROBOTS = 2  # one presser + one marker of the likely receiver


def _split_press_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Split_shape's possession split, with a 2-robot press in place of the lone chaser.

    Without the ball edge and with "press" applicable (an enemy within press range
    of the ball), the first two robots in carrier_first order press and the rest
    defend. Otherwise this is `_possession_split_picker`'s split unchanged.

    Only emits keys for slots it assigns robots to, never a zero-robot entry.
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}

    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    friendly_has_ball_edge = friendly_dist < enemy_dist

    if "press" in available_tactic_ids and not friendly_has_ball_edge:
        partition = {"press": frozenset(ordered[:_PRESS_ROBOTS])}
        rest = ordered[_PRESS_ROBOTS:]
        if rest:
            if "defense" in available_tactic_ids:
                partition["defense"] = frozenset(rest)
            elif "attack" in available_tactic_ids:
                partition["attack"] = frozenset(rest)
        return partition

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


def build_split_press_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) split-press `Strategy` factory.

    Wires `LeadAndSupportTactic` ("attack"), `ShadowAndMarkTactic` ("defense") and
    `PressAndContainTactic` ("press", applicable only within press range of the
    ball), split by `_split_press_picker`.

    Returns a `build_kernel_strategy(motion_controller)` callable suitable for
    `AbstractStrategy`'s constructor argument of the same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": LeadAndSupportTactic(),
                "defense": ShadowAndMarkTactic(),
                "press": PressAndContainTactic(),
            },
            partitioner=_split_press_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
