"""The `press_and_pass` kernel strategy: `build_press_and_pass_kernel_strategy` and the pickers only it uses."""

from __future__ import annotations

from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.pickers import _carrier_first
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic


def _press_and_pass_split_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Same possession-edge split as `_possession_split_picker`, but must also
    respect `PressAndContainTactic.applicable()` — see design doc §15.

    `Strategy` passes `available_tactic_ids` precisely so a picker doesn't
    have to reconstruct a tactic just to ask it a question the kernel
    already knows the answer to. When pressing isn't applicable (no enemy
    near the ball), there is nothing to defend against, so every free robot
    goes to "attack" instead of leaving "defense" empty for no
    game-theoretic reason — a sensible default, not just a way to dodge the
    kernel's applicability check.
    """
    ordered = _carrier_first(game, free_robots)
    if not ordered:
        return {}

    if "defense" not in available_tactic_ids:
        return {"attack": frozenset(ordered)} if "attack" in available_tactic_ids else {}

    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    friendly_has_ball_edge = friendly_dist < enemy_dist

    if "attack" not in available_tactic_ids:
        return {"defense": frozenset(ordered)}

    attack_count = (len(ordered) + 1) // 2 + 1 if friendly_has_ball_edge else len(ordered) // 2 - 1
    attack_count = max(0, min(len(ordered), attack_count))

    partition = {}
    if attack_count > 0:
        partition["attack"] = frozenset(ordered[:attack_count])
    if attack_count < len(ordered):
        partition["defense"] = frozenset(ordered[attack_count:])
    return partition


def build_press_and_pass_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory exercising the newer tactics.

    Wires `GiveAndGoTactic` ("attack") and `PressAndContainTactic`
    ("defense") as two concurrently active slots, split by
    `_press_and_pass_split_picker`. Sibling to
    `build_split_shape_kernel_strategy`, same shape, different tactic pair —
    lets the give-and-go/press-and-contain tactics be driven end to end via
    `StrategyRunner` instead of only unit-level `tick()` calls.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": GiveAndGoTactic(), "defense": PressAndContainTactic()},
            partitioner=_press_and_pass_split_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
