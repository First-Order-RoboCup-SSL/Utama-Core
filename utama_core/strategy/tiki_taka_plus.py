"""The `tiki_taka_plus` kernel strategy: `build_tiki_taka_plus_kernel_strategy` and the pickers only it uses."""

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
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic

# ---------------------------------------------------------------------------
# tiki_taka_plus (2026-08-23 addition)
#
# Every "beat tiki_taka" attempt in the catalog (`overload_press`,
# `high_line_zone`, `counter_flow`) targeted the same weakness: tiki_taka's
# defensive cover is a flat 2 robots in *every* posture, with no
# specialization anywhere on the pitch. The two attempts that tried to
# outnumber that thin line lost before the numbers edge ever mattered --
# tiki_taka's own 3-robot press smothered them in the transition window
# first. `counter_flow` won instead by matching tiki_taka's engine
# (`GiveAndGoTactic`, same press numbers) and adding a wrinkle tiki_taka
# itself lacks (a space-denying block screen instead of man-marking).
#
# `tiki_taka_plus` applies that same lesson *to* tiki_taka rather than
# *against* it: keep tiki_taka's proven build-up/press engine untouched, and
# add the one wrinkle it doesn't have -- a distinct final-third finishing
# move. `_zone_flow_picker` already validated handing the attacking slot
# from `GiveAndGoTactic` to `DecoyOverloadTactic`'s two-role duet once the
# ball crosses into the final third (see `build_zone_fluid_kernel_strategy`);
# this grafts that same handoff onto tiki_taka's stronger 3/2 base shape
# instead of `zone_fluid`'s weaker one.
#
# A second change was considered -- giving the 2 cover robots press
# applicability so a ball breaking toward them can be contested directly
# instead of only shadowed. Investigation of `PressAndContainTactic.
# applicable()` (utama_core/tactics/press_and_contain.py) found this is
# already a no-op: `applicable()` is a global range check against the
# ball-nearest enemy, not scoped to which robots are assigned the "press"
# slot, and `_tiki_taka_picker`'s existing losing+press_ok branch already
# presses with 3 robots (not just the 2 original cover robots) any time the
# ball is lost. tiki_taka's cover pair already gets folded into the press
# the moment possession is lost -- there is no separate "poaching" gap to
# close at the picker level. So `tiki_taka_plus` only implements the
# final-third handoff; it reuses `_tiki_taka_picker`'s possession-edge and
# press logic unchanged.
# ---------------------------------------------------------------------------


def _tiki_taka_plus_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Tiki-taka posture with a final-third finishing wrinkle: possession
    attack with give-and-go in the own/middle thirds, a decoy/overload duet
    once the ball reaches the final third, press on loss, shadow-and-mark
    cover throughout.

    Identical to `_tiki_taka_picker` in every branch except one: when we have
    the ball and it is in the final third, the attacking slot goes to
    `DecoyOverloadTactic` (2 robots, luring the last line out of position)
    instead of `GiveAndGoTactic` (3 robots), with the freed-up robot joining
    cover -- the same zone-conditioned swap `_zone_flow_picker` already
    validated, applied on top of tiki_taka's stronger 3/2 base split rather
    than replacing it everywhere. Losing-possession and non-final-zone
    branches are unchanged from `_tiki_taka_picker`: 3 press + 2 shadow when
    the ball is lost (or all-shadow/all-press if only one of those slots is
    available), 3 attack + 2 cover in the own/mid thirds.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True  # enemy closer, or unknown -> conservative

    press_ok = "press" in available_tactic_ids
    attack_ok = "attack" in available_tactic_ids
    defense_ok = "defense" in available_tactic_ids
    overload_ok = "overload" in available_tactic_ids

    if losing and press_ok:
        if defense_ok:
            return _allocate_ordered(ordered, "press", 3, "defense")
        return _allocate_ordered(ordered, "press", len(ordered))
    if losing:
        if defense_ok:
            return _allocate_ordered(ordered, "defense", len(ordered))
        if attack_ok:
            return _allocate_ordered(ordered, "attack", len(ordered))
        return {}

    # We have the ball: final third hands off to the overload duet, else the
    # give-and-go trio builds up, both keeping a covering pair behind them.
    zone = _ball_zone(game)
    if zone == "final" and overload_ok:
        return _allocate_ordered(ordered, "overload", 2, "defense" if defense_ok else None)
    if attack_ok:
        return _allocate_ordered(ordered, "attack", 3, "defense" if defense_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if defense_ok:
        return _allocate_ordered(ordered, "defense", len(ordered))
    return {}


def build_tiki_taka_plus_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """`tiki_taka` with a final-third finishing wrinkle: same give-and-go
    build-up and press-on-loss engine, but the attacking slot hands off to
    `DecoyOverloadTactic`'s two-role duet once the ball reaches the final
    third, instead of running the give-and-go trio the whole length of the
    pitch. See the "tiki_taka_plus" comment block above
    `_tiki_taka_plus_picker` for the full design rationale, including why a
    second considered change (press applicability for the cover pair) turned
    out to already be present in tiki_taka's existing structure.

    Four concurrent slots -- `GiveAndGoTactic` ("attack"),
    `DecoyOverloadTactic` ("overload"), `PressAndContainTactic` ("press"),
    and `ShadowAndMarkTactic` ("defense") -- allocated by
    `_tiki_taka_plus_picker`: 3+2 attack/cover in the own/mid thirds, 2+3
    overload/cover in the final third, 3+2 press/cover when the ball is
    lost.

    Promoted into `full_match_tournament.py`'s `competitive` tier 2026-08-23
    on its first decoupled side x kickoff round-robin result (6W-7D over 16
    matches, tying `counter_flow` for most wins and clearly outperforming its
    own base `tiki_taka`, 3W-10D) -- see `docs/strategies.md`'s "tiki_taka_plus
    added" section for the full writeup, including the caveat that one
    40-match run isn't yet enough to rule out variance at this sample size.

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
                "press": PressAndContainTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_tiki_taka_plus_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
