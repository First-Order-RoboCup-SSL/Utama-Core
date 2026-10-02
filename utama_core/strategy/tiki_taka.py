"""The `tiki_taka` kernel strategy: `build_tiki_taka_kernel_strategy` and the pickers only it uses."""

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
)
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


def _tiki_taka_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Tiki-taka posture: possession attack with give-and-go, press on loss,
    shadow-and-mark cover in both postures.

    - We have the ball (friendly closer to it, or unknown): 3 attackers
      (give-and-go trio), 2 defenders (one shadowing pair on the shot line).
    - The opponent has the ball: 3 pressers (1 ball-presser + 2 man-markers)
      and 2 shadowers — the press denies the immediate play while the shadow
      pair keeps the shot line honest behind it.
    - A slot that is unavailable (inapplicable — PressAndContain only, or
      commitment-pinned so it never appears in `available_tactic_ids`) has
      its share folded into the other non-pinned attacker/defender slot, so
      every free robot always lands somewhere.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True  # enemy closer, or unknown -> conservative

    press_ok = "press" in available_tactic_ids
    attack_ok = "attack" in available_tactic_ids
    defense_ok = "defense" in available_tactic_ids

    if losing and press_ok:
        # Press with the ball-side group, shadow with the rest; if there is
        # nothing to shadow behind (no defense slot), press with everyone.
        if defense_ok:
            return _allocate_ordered(ordered, "press", 3, "defense")
        return _allocate_ordered(ordered, "press", len(ordered))
    if losing:
        # No press available: everyone shadows/marks.
        if defense_ok:
            return _allocate_ordered(ordered, "defense", len(ordered))
        if attack_ok:
            return _allocate_ordered(ordered, "attack", len(ordered))
        return {}

    # We have the ball: attack with the forward group, keep a covering pair.
    if attack_ok:
        return _allocate_ordered(ordered, "attack", 3, "defense" if defense_ok else None)
    if defense_ok:
        return _allocate_ordered(ordered, "defense", len(ordered))
    return {}


def build_tiki_taka_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Real possession-play team: give-and-go build-up, press on loss,
    shadow-and-mark cover behind every attack.

    Three concurrent slots — `GiveAndGoTactic` ("attack"),
    `PressAndContainTactic` ("press", only when an enemy is within pressing
    range of the ball — its `applicable()`), and
    `ShadowAndMarkTactic` ("defense") — allocated by `_tiki_taka_picker` on
    the possession edge: 3+2 attack/cover when the ball is ours, 3+2
    press/cover when it is lost. The give-and-go loop is the build-up engine:
    short hops under pressure instead of `PassAndShootTactic`'s scripted
    setup-then-shoot, which the match-stuck investigation showed cannot
    complete under contest.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": GiveAndGoTactic(),
                "press": PressAndContainTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_tiki_taka_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
