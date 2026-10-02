"""The `clear_press_plus` kernel strategy: `build_clear_press_plus_kernel_strategy` and the pickers only it uses."""

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
    _clearer_first,
    _friendly_closer_to_ball,
)
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.clear_ball import ClearBallTactic
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic

# ---------------------------------------------------------------------------
# clear_press_plus (2026-09-02 addition)
#
# Stacks the catalog's two most validated single-wrinkle additions onto one
# base instead of picking between them: `clear_danger`'s safety valve (a
# deep-and-contested own-third clearance, the one situational gap no other
# strategy fills) plus `tiki_taka_plus`'s final-third overload handoff (the
# only other single-wrinkle addition with a confirmed independent win --
# 10W-6D in its most recent tournament, clearing the field outright). The
# two wrinkles fire in disjoint game states by construction (the valve needs
# ball-deep-in-our-own-third-and-contested; the overload handoff needs
# ball-in-the-final-third-and-ours), so stacking them should not create a
# new conflict neither wrinkle's own validation run ever exercised.
# ---------------------------------------------------------------------------


def _clear_press_plus_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`_clear_danger_picker`'s allocation, with `_tiki_taka_plus_picker`'s
    final-third handoff grafted onto its attacking branch: when we hold the
    ball and it is not deep-and-contested in our own third, the final third
    hands off from the give-and-go trio to the 2-robot overload duet instead
    of running give-and-go the length of the pitch. The clearance valve and
    the losing-possession press/block branches are unchanged from
    `_clear_danger_picker`.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    clear_ok = "clear" in available_tactic_ids
    press_ok = "press" in available_tactic_ids
    block_ok = "block" in available_tactic_ids
    attack_ok = "attack" in available_tactic_ids
    overload_ok = "overload" in available_tactic_ids

    if clear_ok:
        if len(ordered) == 1 or not block_ok:
            return {"clear": frozenset(ordered)}
        clearer_order = _clearer_first(game, ordered)
        return {"clear": frozenset(clearer_order[:1]), "block": frozenset(clearer_order[1:])}

    currently_attacking = bool(prev_partition.get("attack")) or bool(prev_partition.get("overload"))
    friendly_edge = _friendly_closer_to_ball(game)
    if currently_attacking:
        losing = friendly_edge is False
    else:
        losing = friendly_edge is not True

    if losing:
        if press_ok:
            return _allocate_ordered(ordered, "press", 3, "block" if block_ok else None)
        if block_ok:
            return {"block": frozenset(ordered)}
        if attack_ok:
            return {"attack": frozenset(ordered)}
        return {}

    zone = _ball_zone(game)
    if zone == "final" and overload_ok:
        return _allocate_ordered(ordered, "overload", 2, "block" if block_ok else None)
    if attack_ok:
        return _allocate_ordered(ordered, "attack", 3, "block" if block_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if block_ok:
        return {"block": frozenset(ordered)}
    return {}


def build_clear_press_plus_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """`clear_danger` plus `tiki_taka_plus`'s final-third overload handoff.

    Five concurrent slots -- `GiveAndGoTactic` ("attack"),
    `DecoyOverloadTactic` ("overload"), `ClearBallTactic` ("clear"),
    `PressAndContainTactic` ("press"), and `BlockShapeTactic` ("block") --
    allocated by `_clear_press_plus_picker`: identical to `clear_danger`
    except the attacking branch hands off from give-and-go to the overload
    duet once the ball reaches the final third, the same swap
    `tiki_taka_plus` validated on `tiki_taka`'s base.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": GiveAndGoTactic(),
                "overload": DecoyOverloadTactic(),
                "clear": ClearBallTactic(),
                "press": PressAndContainTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_clear_press_plus_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
