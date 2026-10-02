"""The `score_aware_counter_flow` kernel strategy: `build_score_aware_counter_flow_kernel_strategy` and the pickers only it uses."""

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
    _friendly_score_diff,
    _is_late_in_half,
)
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic

# ---------------------------------------------------------------------------
# score_aware_counter_flow (2026-09-02 addition)
#
# `score_aware_zone_flow` validated the scoreline-aware attack/defense split
# shift as a free improvement grafted onto `zone_fluid`'s picker with zero
# other changes -- but `zone_fluid` is one of the catalog's weaker bases
# (2W-6D-5L in the 2026-08-21 backfill). This strategy applies the identical
# mechanism (shrink the attacking commitment when ahead late, grow it when
# behind late) to `counter_flow`'s picker instead -- the catalog's only
# undefeated strategy, 0L across every backfill/tournament to date. Every
# other axis is left untouched: same 3 tactics (`GiveAndGoTactic`/
# `PressAndContainTactic`/`BlockShapeTactic`), same sticky possession-edge
# hysteresis, same 3/2 split everywhere except the late-game scoreline
# branch -- so any observed difference from plain `counter_flow` is
# attributable to the new scoreline input alone, not a different base.
# ---------------------------------------------------------------------------


def _score_aware_counter_flow_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`_counter_flow_picker`'s allocation, with the attacking commitment size
    shifted late in a half by whether we're ahead or behind -- see
    `_score_aware_zone_flow_picker` for the identical mechanism applied to a
    different base.

    Not late, tied, or score unreadable: identical to `_counter_flow_picker`
    (3 attack/2 block when we hold the edge, 3 press/2 block when we don't).
    Late and ahead: only 2 attack, 3 block -- protect the lead behind a
    heavier screen. Late and behind: 4 attack, 1 block -- chase an
    equaliser. The press branch (losing possession) is untouched by score:
    a team already chasing the ball has nothing to protect or chase by
    committing fewer/more pressers, only by out-scoring once it regains the
    ball, which the attack branch already covers.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    currently_attacking = bool(prev_partition.get("attack"))
    friendly_edge = _friendly_closer_to_ball(game)
    if currently_attacking:
        losing = friendly_edge is False
    else:
        losing = friendly_edge is not True

    attack_ok = "attack" in available_tactic_ids
    press_ok = "press" in available_tactic_ids
    block_ok = "block" in available_tactic_ids

    if losing:
        if press_ok:
            return _allocate_ordered(ordered, "press", 3, "block" if block_ok else None)
        if block_ok:
            return _allocate_ordered(ordered, "block", len(ordered))
        if attack_ok:
            return _allocate_ordered(ordered, "attack", len(ordered))
        return {}

    if attack_ok:
        attackers = 3
        if _is_late_in_half(game):
            score_diff = _friendly_score_diff(game)
            if score_diff is not None and score_diff > 0:
                attackers = 2
            elif score_diff is not None and score_diff < 0:
                attackers = 4
        return _allocate_ordered(ordered, "attack", attackers, "block" if block_ok else None)
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def build_score_aware_counter_flow_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """`counter_flow` (the catalog's only undefeated strategy) with the same
    scoreline-aware attack-commitment shift `score_aware_zone_flow` validated
    on a weaker base. Same three tactics/slots as `counter_flow`
    (`GiveAndGoTactic`/`PressAndContainTactic`/`BlockShapeTactic`) and the
    same sticky possession-edge hysteresis -- only the picker's late-game
    attack-count decision differs. See `_score_aware_counter_flow_picker`.

    Returns a `build_kernel_strategy(motion_controller)` callable suitable
    for `AbstractStrategy`'s constructor argument of the same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": GiveAndGoTactic(),
                "press": PressAndContainTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_score_aware_counter_flow_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
