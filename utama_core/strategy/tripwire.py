"""`tripwire` (until 2026-11-12 `press_trigger_flow`): `riptide` that goes all in on the ball once it is lost in its own third. The pickers only it uses are here."""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.common.motion_controller import MotionController
from utama_core.strategy.pickers import (
    allocate_ordered,
    ball_zone,
    carrier_first,
    friendly_closer_to_ball,
)
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic

# ---------------------------------------------------------------------------
# tripwire (2026-09-02 addition)
#
# `riptide` presses with a fixed 3 robots any time pressing is
# applicable at all, regardless of where on the pitch that press happens --
# the same commitment whether the ball was lost at the halfway line or deep
# in our own third. `ball_zone` is already read by every zone-handoff
# picker (`zone_fluid`, `weaver`, `high_line_zone`, ...) to change
# the *attacking* pattern by zone, but no picker in the catalog uses it to
# change the *press* commitment by zone -- this strategy is that missing
# combination: riptide's proven engine, with the press going all-in
# (every free robot, not just 3) specifically when the loss happens in our
# own third, since a loose ball that close to our own goal is the one
# situation where `firebreak`'s own author-noted lesson ("ball control,
# not territory, is the bottleneck") is most costly to get wrong. Everywhere
# else (mid/final third losses, and every attacking-side branch), this is
# byte-for-byte identical to `_riptide_picker`.
# ---------------------------------------------------------------------------


def _tripwire_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`_riptide_picker`'s allocation, with an all-in press (every free
    robot, not the usual 3+2 split) when the ball is lost specifically in
    our own third. Mid/final-third losses and every attacking-side branch
    are unchanged from `_riptide_picker` -- see that picker's
    docstring for the shared attack/press/block rationale this one inherits
    unmodified outside the own-third-press branch.
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    currently_attacking = bool(prev_partition.get("attack"))
    friendly_edge = friendly_closer_to_ball(game)
    if currently_attacking:
        losing = friendly_edge is False
    else:
        losing = friendly_edge is not True

    attack_ok = "attack" in available_tactic_ids
    press_ok = "press" in available_tactic_ids
    block_ok = "block" in available_tactic_ids

    if losing:
        if press_ok:
            zone = ball_zone(game)
            if zone == "own":
                return allocate_ordered(ordered, "press", len(ordered))
            return allocate_ordered(ordered, "press", 3, "block" if block_ok else None)
        if block_ok:
            return allocate_ordered(ordered, "block", len(ordered))
        if attack_ok:
            return allocate_ordered(ordered, "attack", len(ordered))
        return {}

    if attack_ok:
        return allocate_ordered(ordered, "attack", 3, "block" if block_ok else None)
    if block_ok:
        return allocate_ordered(ordered, "block", len(ordered))
    return {}


def build(outfield_robot_ids: tuple[int, ...], motion_controller: MotionController) -> KernelSchedulerStrategy:
    """`riptide` with a zone-triggered all-in press: the usual 3-press/
    2-block split everywhere, except an all-in press (every free robot) when
    the ball is lost in our own third specifically. See the
    "tripwire" comment block above `_tripwire_picker`
    for the full rationale.

    Same three tactics/slots as `riptide` (`GiveAndGoTactic`/
    `PressAndContainTactic`/`BlockShapeTactic`) -- only the picker's
    own-third press commitment differs, so any observed difference from
    plain `riptide` is attributable to the zone-triggered press alone.

    Returns a `build(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    ctx = TickContext(motion_controller=motion_controller)
    return KernelSchedulerStrategy(
        tactics={
            "attack": GiveAndGoTactic(),
            "press": PressAndContainTactic(),
            "block": BlockShapeTactic(),
        },
        partitioner=_tripwire_picker,
        outfield_robot_ids=outfield_robot_ids,
        ctx=ctx,
    )
