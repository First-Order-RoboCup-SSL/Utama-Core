"""The `shadow_switch` kernel strategy: `build_shadow_switch_kernel_strategy` and the pickers only it uses."""

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
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic
from utama_core.tactics.switch_of_play import SwitchOfPlayTactic

# ---------------------------------------------------------------------------
# shadow_switch (2026-09-02 addition)
#
# `high_line_zone` paired `SwitchOfPlayTactic` with `BlockShapeTactic`'s zone
# screen and stayed parked (2-7-4 in its best backfill) -- the screen denies
# individual matchups, but the switch's carrier/pivot/runner relay still
# needs several seconds to settle, which a genuinely contested match rarely
# grants. `tiki_taka`'s `ShadowAndMarkTactic` defense is independently
# proven viable (it's the top-tested strategy's own defense, 1W-8D-4L in
# isolation but never the bottleneck -- every loss traced to the *attack*
# side, not the shadow cover). This pairing -- switch's attack, shadow's
# defense -- has never been tried: `high_line_zone` used switch with the
# zone screen, `tiki_taka`/`tiki_taka_plus` used shadow with give-and-go.
# Hypothesis: man-marking cover (proven to hold up on its own) may free the
# switch attack to develop without the zone screen's own unproven defensive
# contribution as a confound in either direction.
# ---------------------------------------------------------------------------


def _shadow_switch_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Switch-attack, shadow-defense posture.

    - The opponent has the ball: the whole team shadows/marks
      (`ShadowAndMarkTactic`), same defensive shape `tiki_taka` already
      validates.
    - We have the ball: `SwitchOfPlayTactic` leads with 3 (the relay's
      carrier/pivot/runner roles), 2 hold shadow cover behind it -- same
      3/2 split shape as `high_line_zone`'s switch branch, defense slot
      swapped.

    Sticky possession edge, same fix `_high_line_zone_picker`/
    `_counter_flow_picker` needed: the switch relay needs several seconds to
    settle, which a tick-by-tick possession-edge re-read does not reliably
    give it in a contested match. `prev_partition` ("did we hold switch last
    tick") is the memory; require a clear possession loss to drop it.
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    currently_attacking = bool(prev_partition.get("switch"))
    friendly_edge = friendly_closer_to_ball(game)
    if currently_attacking:
        losing = friendly_edge is False
    else:
        losing = friendly_edge is not True

    switch_ok = "switch" in available_tactic_ids
    defense_ok = "defense" in available_tactic_ids

    if losing:
        if defense_ok:
            return allocate_ordered(ordered, "defense", len(ordered))
        if switch_ok:
            return allocate_ordered(ordered, "switch", len(ordered))
        return {}

    if switch_ok:
        return allocate_ordered(ordered, "switch", 3, "defense" if defense_ok else None)
    if defense_ok:
        return allocate_ordered(ordered, "defense", len(ordered))
    return {}


def build_shadow_switch_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Switch-of-play attack with man-marking cover -- a tactic pairing never
    tried elsewhere in the catalog (`high_line_zone` paired the switch with
    a zone screen; `tiki_taka`/`tiki_taka_plus` paired shadow-and-mark with
    give-and-go). See the "shadow_switch" comment block above
    `_shadow_switch_picker` for the full rationale.

    Two concurrent slots -- `SwitchOfPlayTactic` ("switch") and
    `ShadowAndMarkTactic` ("defense") -- allocated by `_shadow_switch_picker`
    on a 3/2 split with sticky possession-edge hysteresis, matching the
    settling time the switch relay needs.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "switch": SwitchOfPlayTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_shadow_switch_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
