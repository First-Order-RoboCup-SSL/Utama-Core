"""The `overload_flow` kernel strategy: `build_overload_flow_kernel_strategy` and the pickers only it uses."""

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

# ---------------------------------------------------------------------------
# overload_flow (2026-09-02 addition)
#
# Every existing possession-edge picker (including `_zone_flow_picker`,
# which this strategy is otherwise identical to) reacts to a single tick's
# edge read alone. `score_aware_zone_flow` proved a cheap graft onto that
# same base is worth testing (the scoreline); this strategy grafts a
# different, still-unused signal instead: how *long* the possession edge has
# held, not just its current value. Rationale: `_zone_flow_picker`'s
# give-and-go trio already builds patiently through the middle thirds --
# committing a 4th attacker the instant the edge flips, before that
# possession is actually secure, risks overextending into exactly the kind
# of single-tick noise `_CLOSER_TO_BALL_MARGIN` and every sticky-edge picker
# (`counter_flow`, `high_line_zone`, `shadow_switch`) already had to guard
# against elsewhere. Only commit the extra body once the edge has held for a
# real, sustained window.
# ---------------------------------------------------------------------------

# ~1.5s at the kernel's tick rate (60Hz, matching every other tick-count
# constant in this file, e.g. `PressAndContainTactic`'s hysteresis window)
# -- long enough to filter a
# single contested-ball flicker, short enough that a genuine sustained
# possession spell still gets the extra attacker well before a give-and-go
# hop cycle completes.
_POSSESSION_STREAK_TICKS = 90


# Consecutive-possession-tick counter, keyed by nothing (a single team only
# ever runs one `_overload_flow_picker` instance per process, unlike
# `shielding.py`'s per-robot-id state) -- a `Partitioner` is a plain
# function with no `mem` of its own, and `prev_partition`'s type is fixed to
# `dict[str, frozenset[RobotId]]`, which cannot carry a tick count without
# inventing a fake tactic id (rejected: `Strategy._validate_partition`/
# `_slot_for` both raise `KeyError` on any key absent from the tactic
# registry -- confirmed by reading both directly -- so a smuggled key would
# crash on tick 1, not silently no-op). Module-level state, same pattern
# `shielding.py`'s `_COMMITTED_ROBOTS` already establishes for picker/skill
# memory the kernel's own plumbing has no channel for.
_possession_streak: int = 0


def _overload_flow_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`_zone_flow_picker`'s allocation, with the own/mid-third give-and-go
    attacker count growing from 3 to 4 only once the possession edge has
    held for `_POSSESSION_STREAK_TICKS` consecutive ticks, instead of
    reacting to the edge's current value alone.

    Not late in a possession spell, losing, or in the final third:
    identical to `_zone_flow_picker`. Final-third handoff to the overload
    duet is unchanged (that tactic is a 2-role duet regardless of streak
    length -- the streak only ever affects the own/mid-third give-and-go
    count).
    """
    global _possession_streak
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True
    _possession_streak = 0 if losing else min(_possession_streak + 1, _POSSESSION_STREAK_TICKS)
    streak = _possession_streak

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
        attackers = 4 if streak >= _POSSESSION_STREAK_TICKS else 3
        return _allocate_ordered(ordered, "givego", attackers, "defense" if defense_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if defense_ok:
        return _allocate_ordered(ordered, "defense", len(ordered))
    return {}


def build_overload_flow_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """`zone_fluid` with a possession-*duration*-aware give-and-go attacker
    count: 3 attackers as usual, growing to 4 only once the possession edge
    has held for a sustained window (`_POSSESSION_STREAK_TICKS`), instead of
    reacting to a single tick's edge read the way every other picker in the
    catalog does. See the "overload_flow" comment block above
    `_overload_flow_picker` for the full rationale.

    Same three tactics/slots as `zone_fluid` (`GiveAndGoTactic`/
    `DecoyOverloadTactic`/`ShadowAndMarkTactic`) -- only the picker's
    attacker-count decision differs, so any observed difference from plain
    `zone_fluid` is attributable to the streak requirement alone.

    Returns a `build_kernel_strategy(motion_controller)` callable suitable
    for `AbstractStrategy`'s constructor argument of the same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        # `_possession_streak` is module-level, not per-`Strategy` state (see
        # its own docstring for why) -- a tournament worker process reuses
        # its process across many matches (`round_robin.py`'s
        # `ProcessPoolExecutor`), so without this reset a match would start
        # with whatever streak count the *previous* match on this worker
        # ended at, silently giving `overload_flow` a false head start (or a
        # false handicap) at kickoff. `_build` runs exactly once per match
        # (`AbstractStrategy.__init__` calls it fresh every time), making
        # this the natural per-match reset point.
        global _possession_streak
        _possession_streak = 0
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "givego": GiveAndGoTactic(),
                "overload": DecoyOverloadTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_overload_flow_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
