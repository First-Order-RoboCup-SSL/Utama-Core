"""The `counter_flow` kernel strategy: `build_counter_flow_kernel_strategy` and the pickers only it uses."""

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
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic


def _counter_flow_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Counter-flow posture: fight tiki_taka on its own strongest ground —
    beat its give-and-go attack with our own give-and-go, contest its press
    window with our own press instead of ceding it, and fall back to a
    space-denying screen (not man-shadow) when neither side has an edge.

    Both `overload_press` and `high_line_zone` tried to out-number
    tiki_taka's 2-robot shadow defense and lost before that numbers edge
    ever got to matter — tiki_taka's 3-robot press (1 presser + 2 markers,
    `PressAndContainTactic` only becomes applicable within 1.5 m of the
    ball) smothered both attempts in the transition window, every time. The
    lesson from that failure isn't "the overload theory was wrong" — it's
    that neither prior attempt had a real answer to the press itself:
    `LeadAndSupportTactic` never passes at all (dribble-only, easy to
    press-trap), and the switch-of-play relay needs several seconds to
    settle that the press never gave it. `GiveAndGoTactic` is the one
    attacking tactic in the catalog actually built to survive exactly this
    — cycle a fresh hop the instant the current lane closes, shoot the
    moment one opens — and it is tiki_taka's *own* attack engine, already
    proven undefeated. This posture borrows it rather than assuming our own
    novel attack pattern would fare better than tiki_taka's did.

    - We have the ball (or unknown): 3 attack (give-and-go trio, matching
      tiki_taka's own commitment there so we are not thinner in possession
      than the team we are trying to out-cycle), 2 hold the block screen —
      space denial instead of `ShadowAndMarkTactic`'s greedy man-marking,
      so committing bodies forward never leaves us exposed to the
      clustering/defense-area fouls a shadow line invites under pressure.
    - The opponent has the ball and is pressable: press with 3 (matching
      tiki_taka's own press numbers instead of ceding the transition
      window to it, which is exactly the window both prior attempts lost
      in), block screen with the rest.
    - The opponent has the ball but is not pressable (out of press range):
      the whole team holds the block screen — no reason to commit anyone
      forward with no ball-side trigger.

    Sticky possession edge, same fix `_high_line_zone_picker` needed: a
    plain re-read of `_friendly_closer_to_ball` every tick flips constantly
    in a genuinely contested match (measured on the first real run against
    tiki_taka: attack/press alternated every 1-3 s for the entire first
    7 s, `GiveAndGoTactic`'s hop-cycle reset before a single hop completed,
    6%/94% possession, 4.8 m of ball travel in 60 s). `prev_partition` is
    this picker's only persistent state, so use "did we hold attack last
    tick" as memory and require a clear loss (not just "not clearly ahead")
    before giving it up — mirrors `_high_line_zone_picker`'s identical fix.
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
        return _allocate_ordered(ordered, "attack", 3, "block" if block_ok else None)
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def build_counter_flow_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Direct answer to tiki_taka: our own give-and-go attack, our own press
    on the ball, a space-denying screen everywhere else.

    Three concurrent slots — `GiveAndGoTactic` ("attack"),
    `PressAndContainTactic` ("press"), and `BlockShapeTactic` ("block") —
    allocated by `_counter_flow_picker` on a 3/2 split matching tiki_taka's
    own commitment in both postures, rather than the numbers-overload bet
    `overload_press`/`high_line_zone` made and lost. The theory: neither
    prior anti-tiki_taka attempt ever got far enough to test its numbers
    edge because tiki_taka's press won the transition window outright: this
    strategy contests that window symmetrically (our own 3-press) instead
    of trying to dodge it, and uses tiki_taka's own proven-undefeated attack
    engine rather than a fresh pattern.

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
                "block": BlockShapeTactic(),
            },
            partitioner=_counter_flow_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
