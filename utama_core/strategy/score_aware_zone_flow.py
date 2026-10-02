"""The `score_aware_zone_flow` kernel strategy: `build_score_aware_zone_flow_kernel_strategy` and the pickers only it uses."""

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
    _friendly_score_diff,
    _is_late_in_half,
)
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic

# ---------------------------------------------------------------------------
# Score-aware zone-flow (2026-08-23 addition)
#
# Every existing picker reads possession/ball-zone but none reads the
# scoreline itself, even though `game.referee.{yellow,blue}_team.score` is
# already populated in any refereed match (`full_match_tournament.py`/
# `round_robin.py` already read the same fields to report a match's result).
# This variant is `_zone_flow_picker`'s allocation with one added axis: late
# in a half, shift the *size* of the attack/defense split based on whether
# we're ahead or behind, instead of holding the same 3/2-ish split regardless
# of score. Ahead late: fewer bodies committed forward, more men behind the
# ball to protect the lead. Behind late: the reverse, chase the game. Tied,
# early in the half, or score unreadable (`game.referee is None`): identical
# to `_zone_flow_picker`, so this strategy only diverges from `zone_fluid`
# when the new signal actually has something to say.
# ---------------------------------------------------------------------------


def _score_aware_zone_flow_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`_zone_flow_picker`'s allocation, with the attack/defense split size
    shifted late in a half by whether we're ahead or behind.

    Not late, tied, or score unreadable: identical to `_zone_flow_picker`
    (3 attackers/2 defenders when we have the ball, everyone shadows when we
    don't). Late and ahead: only 2 forward, 3 back, to protect the lead. Late
    and behind: 4 forward, 1 back, to chase an equaliser. The final-third
    overload duet always stays 2 robots regardless (that tactic is a
    two-role duet by design, see `_zone_flow_picker`) — only the
    own/mid-third give-and-go split size reacts to the scoreline.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing_possession = friendly_edge is not True

    defense_ok = "defense" in available_tactic_ids
    givego_ok = "givego" in available_tactic_ids
    overload_ok = "overload" in available_tactic_ids

    if losing_possession:
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
        attackers = 3
        if _is_late_in_half(game):
            score_diff = _friendly_score_diff(game)
            if score_diff is not None and score_diff > 0:
                attackers = 2
            elif score_diff is not None and score_diff < 0:
                attackers = 4
        return _allocate_ordered(ordered, "givego", attackers, "defense" if defense_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if defense_ok:
        return _allocate_ordered(ordered, "defense", len(ordered))
    return {}


def build_score_aware_zone_flow_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Zone-flow team (see `build_zone_fluid_kernel_strategy`) with one added
    decision input nothing else in the catalog uses: the scoreline. Late in a
    half, the give-and-go attack/defense split shrinks when we're ahead
    (protect the lead) and grows when we're behind (chase the game), instead
    of holding the same split regardless of score. Same three tactics/slots
    as `zone_fluid` (`GiveAndGoTactic`/`DecoyOverloadTactic`/
    `ShadowAndMarkTactic`) — only the picker differs, so any difference in
    results is attributable to the new score-aware allocation, not a
    different tactic set.

    Returns a `build_kernel_strategy(motion_controller)` callable suitable
    for `AbstractStrategy`'s constructor argument of the same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "givego": GiveAndGoTactic(),
                "overload": DecoyOverloadTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_score_aware_zone_flow_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
