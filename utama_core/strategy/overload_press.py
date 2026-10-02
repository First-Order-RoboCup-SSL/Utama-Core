"""The `overload_press` kernel strategy: `build_overload_press_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import (
    allocate_ordered,
    carrier_first,
    friendly_closer_to_ball,
)
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.switch_of_play import SwitchOfPlayTactic

# ---------------------------------------------------------------------------
# Anti-tiki_taka strategies (2026-08-20 addition)
#
# tiki_taka (`_tiki_taka_picker`) splits 3/2 in both postures: 3 attack + 2
# defense when it has the ball, 3 press + 2 defense when it doesn't. Two
# exploitable properties fall directly out of that split:
#
# 1. Its "defense" slot is `ShadowAndMarkTactic`, whose man-marking only
#    starts at the *3rd* assigned robot (robots 1-2 always shadow the shot
#    line; see the tactic's own docstring/tick — marking is
#    `robot_ids[2:]`). tiki_taka only ever gives that slot 2 robots, in
#    either posture — so its defense is *always* pure shot-line shadowing,
#    never man-marking, no matter how many attackers we send. A numbers
#    overload (more attacking bodies than tiki_taka has cover for) faces
#    zero marking, only a two-robot shadow to beat with width or a switch.
# 2. Its 3-press only forms *after* it reads possession loss — there is no
#    press while it still has the ball, and nothing pre-positioned for a
#    turnover. A fast direct counter (no scripted setup phase, re-picks its
#    leader every uncommitted tick) that strikes in the transition window
#    before the 3-press organizes skips the fight tiki_taka is built to win.
# ---------------------------------------------------------------------------


def _overload_press_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Overload-and-strike posture: outnumber tiki_taka's 2-robot shadow line
    when we have the ball, hit immediately on a turnover before its press
    organizes.

    - We have the ball (or unknown): 4 robots overload (`DecoyOverloadTactic`
      lure + fill, backed by `SwitchOfPlayTactic`'s weak-side read once the
      overload draws cover across) — more attackers than tiki_taka's defense
      slot ever man-marks, since that slot never grows past 2 and only
      shadows. 1 robot holds `BlockShapeTactic` as counter insurance.
    - The opponent has the ball: everyone (or as many as `available_tactic_ids`
      allows) goes straight to `LeadAndSupportTactic` — a direct,
      no-setup-phase counter — rather than a organized press, to strike in
      the transition window before tiki_taka's own 3-press forms. Falls back
      to the block screen if the counter is unavailable (e.g. `switch`
      pinned mid-relay elsewhere — not expected with this tactic set, but
      every picker in this file keeps this fallback chain for the same
      reason: every free robot must land somewhere).
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}

    friendly_edge = friendly_closer_to_ball(game)
    losing = friendly_edge is not True

    overload_ok = "overload" in available_tactic_ids
    switch_ok = "switch" in available_tactic_ids
    block_ok = "block" in available_tactic_ids
    counter_ok = "counter" in available_tactic_ids

    if losing:
        if counter_ok:
            return allocate_ordered(ordered, "counter", len(ordered))
        if block_ok:
            return allocate_ordered(ordered, "block", len(ordered))
        if switch_ok:
            return allocate_ordered(ordered, "switch", len(ordered))
        return {}

    if overload_ok:
        return allocate_ordered(ordered, "overload", 4, "block" if block_ok else None)
    if switch_ok:
        return allocate_ordered(ordered, "switch", 4, "block" if block_ok else None)
    if block_ok:
        return allocate_ordered(ordered, "block", len(ordered))
    return {}


def build_overload_press_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Numbers-overload team built to beat tiki_taka's shot-line-only shadow
    defense, with a direct pre-press counter for the transition window.

    Four concurrent slots — `DecoyOverloadTactic` ("overload"),
    `SwitchOfPlayTactic` ("switch"), `LeadAndSupportTactic` ("counter"), and
    `BlockShapeTactic` ("block") — allocated by `_overload_press_picker`:
    4 robots overload/switch the attack (more bodies than tiki_taka's 2-robot
    defense slot ever marks) with 1 held on the block screen while we have
    the ball; on loss, the whole team goes direct via the counter rather than
    building a press, to hit before tiki_taka's own 3-press organizes.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "overload": DecoyOverloadTactic(),
                "switch": SwitchOfPlayTactic(),
                "counter": LeadAndSupportTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_overload_press_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
