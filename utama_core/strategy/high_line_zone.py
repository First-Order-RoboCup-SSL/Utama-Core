"""The `high_line_zone` kernel strategy: `build_high_line_zone_kernel_strategy` and the pickers only it uses."""

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
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.switch_of_play import SwitchOfPlayTactic


def _high_line_zone_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """High-line zone posture: deny tiki_taka's give-and-go trio the 1v1s it
    wants with a zone screen instead of man-marking, switch the ball to the
    weak side its 2-robot defense can't cover, finish with the overload duet
    once we're through.

    - The opponent has the ball: the whole team holds `BlockShapeTactic`'s
      shifting zone line — tiki_taka's give-and-go build-up is a 3-robot
      short-passing relay that thrives against man-marking cover it can
      dribble/pass past one defender at a time; a zone screen that shifts
      with the ball and never breaks shape denies it the individual
      matchups it's built around, unlike `ShadowAndMarkTactic`'s greedy
      per-robot marking (which is what tiki_taka's own defense runs, and
      exactly the shape `overload_press` targets instead).
    - We have the ball, not yet in the final third: `SwitchOfPlayTactic`
      leads — tiki_taka's defense is only ever 2 robots, so a genuine
      weak-side imbalance is easy to manufacture; 2 hold the block screen as
      insurance against the counter.
    - We have the ball in the final third: switch to the overload duet
      (`DecoyOverloadTactic`) to finish, same final-third handoff
      `_zone_flow_picker` uses — the switch has already done its job of
      breaking the defense's shape open by this point.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    block_ok = "block" in available_tactic_ids
    switch_ok = "switch" in available_tactic_ids
    overload_ok = "overload" in available_tactic_ids

    # Sticky possession edge: a plain `_friendly_closer_to_ball` re-read every
    # tick flips constantly in a genuinely contested 50/50 (measured: switch
    # assigned and released again within single-digit ticks, over and over,
    # for the first ~10s of a live match against tiki_taka) — `switch`'s
    # carrier/pivot/runner relay needs several seconds to settle and never
    # got the chance, discarded before `is_committed()` ever saw it commit.
    # `prev_partition` is this picker's only persistent state (a `Partitioner`
    # is a plain function, no `mem` of its own — unlike a `Tactic`, which
    # gets one), so use "did we hold switch/overload last tick" as the
    # attacking side's memory and require a full possession loss (proximity
    # edge, not just "not clearly ahead") before dropping it. Mirrors
    # `SwitchOfPlayTactic`'s own internal weak-side hysteresis, one level up.
    prev_partition = prev_partition or {}
    currently_attacking = bool(prev_partition.get("switch")) or bool(prev_partition.get("overload"))
    friendly_edge = _friendly_closer_to_ball(game)
    if currently_attacking:
        losing = friendly_edge is False  # only give up the ball on a clear loss, not just "unknown"
    else:
        losing = friendly_edge is not True  # regaining needs a clear win, same conservative default as elsewhere

    if losing:
        if block_ok:
            return _allocate_ordered(ordered, "block", len(ordered))
        if switch_ok:
            return _allocate_ordered(ordered, "switch", len(ordered))
        if overload_ok:
            return _allocate_ordered(ordered, "overload", len(ordered))
        return {}

    zone = _ball_zone(game)
    if zone == "final" and overload_ok:
        return _allocate_ordered(ordered, "overload", 2, "block" if block_ok else None)
    if switch_ok:
        return _allocate_ordered(ordered, "switch", 3, "block" if block_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def build_high_line_zone_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Zone-defense team built to deny tiki_taka's give-and-go trio the 1v1s
    it's designed around, switching the ball past its thin 2-robot cover.

    Three concurrent slots — `SwitchOfPlayTactic` ("switch"),
    `DecoyOverloadTactic` ("overload"), and `BlockShapeTactic` ("block") —
    allocated by `_high_line_zone_picker`: the whole team holds the zone
    screen when the ball is lost (denying man-marking 1v1s instead of
    running them, unlike tiki_taka's own `ShadowAndMarkTactic` defense);
    3 robots read the weak side and switch there once we have it (tiki_taka's
    defense is always just 2 robots — an imbalance is easy to create); the
    final third hands off to the overload duet to finish.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "switch": SwitchOfPlayTactic(),
                "overload": DecoyOverloadTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_high_line_zone_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
