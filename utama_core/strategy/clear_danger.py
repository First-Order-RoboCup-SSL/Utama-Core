"""The `clear_danger` kernel strategy: `build_clear_danger_kernel_strategy` and the pickers only it uses."""

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
    _clearer_first,
    _friendly_closer_to_ball,
)
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.clear_ball import ClearBallTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic


def _clear_danger_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Safety-valve posture: counter_flow's proven attack/press/block postures,
    plus a danger valve — when the ball is deep in our own third AND contested
    (`ClearBallTactic.applicable()`, the real §15 gate this config exists to
    exercise), one robot clears it long while everyone else holds the screen.

    - Danger: 1 clearer + block screen with the rest. The valve overrides both
      other postures: building out of a contested own-third pocket is exactly
      what `GiveAndGoTactic` should not be doing (its hops are short by design),
      and pressing leaves nobody removing the ball.
    - Lost, no danger: counter_flow's defensive posture — press 3 + block 2
      when something is pressable, all-block otherwise.
    - Won: counter_flow's attacking posture — attack 3 + block 2.

    The press slot is NOT optional here (an earlier draft dropped it in favour
    of block-everyone-unless-danger): traced live, that team never contests any
    ball anywhere — possession collapsed to ~1% as the opponent's carrier parked
    unpressured at midfield, and the ball never even reached our third, so the
    valve itself never fired. Ball control, not territory, is the bottleneck
    (the same lesson as `go_to_ball`'s shield fix); counter_flow's press stays,
    the valve rides on top of it.

    Sticky possession edge, same fix `_counter_flow_picker`/`_high_line_zone_picker`
    needed: re-reading `_friendly_closer_to_ball` every tick flips constantly in
    a genuinely contested match and resets `GiveAndGoTactic`'s hop cycle before
    a single hop completes. `prev_partition` ("did we hold attack last tick") is
    the memory; require a clear possession loss to drop it.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    clear_ok = "clear" in available_tactic_ids
    press_ok = "press" in available_tactic_ids
    block_ok = "block" in available_tactic_ids
    attack_ok = "attack" in available_tactic_ids

    if clear_ok:
        # The valve takes exactly 1 robot (the carrier, else the one nearest the
        # ball); everything else holds the screen. A pinned/inapplicable block
        # slot folds its share into the valve slot — every free robot must land
        # somewhere.
        if len(ordered) == 1 or not block_ok:
            return {"clear": frozenset(ordered)}
        clearer_order = _clearer_first(game, ordered)
        return {"clear": frozenset(clearer_order[:1]), "block": frozenset(clearer_order[1:])}

    currently_attacking = bool(prev_partition.get("attack"))
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

    if attack_ok:
        return _allocate_ordered(ordered, "attack", 3, "block" if block_ok else None)
    if block_ok:
        return {"block": frozenset(ordered)}
    return {}


def build_clear_danger_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Safety-valve team: counter_flow's postures plus a danger-clearance valve.

    Four concurrent slots — `GiveAndGoTactic` ("attack"),
    `ClearBallTactic` ("clear"), `PressAndContainTactic` ("press"), and
    `BlockShapeTactic` ("block") — allocated by `_clear_danger_picker`.
    Identical to `counter_flow` whenever the ball is not deep-and-contested in
    our own third; when it is, the valve takes one robot and removes the ball
    instead of only screening it. Fills the one hole the catalog has: no
    existing tactic ever deliberately kicks the ball out of danger (the only
    kick sites aim at the enemy goal), so pinned-deep teams can only
    shadow/mark/block and wait.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": GiveAndGoTactic(),
                "clear": ClearBallTactic(),
                "press": PressAndContainTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_clear_danger_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
