"""WanderingTactic — base kernel tactic for referee visualisation.

Each robot cycles through its own list of waypoints on the field indefinitely.
When a referee command fires, `kernel.Strategy.tick()`'s `RefereeOverride`
intercepts before this tactic runs, so you can clearly see robots interrupted
and repositioned by the referee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.kernel.abstract_strategy import AbstractStrategy
from utama_core.kernel.context import KernelContext
from utama_core.kernel.strategy import Strategy as KernelSchedulerStrategy
from utama_core.kernel.tactic import BaseTactic, RobotId, TacticTag
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move

# Waypoints defined as fractions of the standard field half-dimensions
# (half_length=4.5, half_width=3.0) so they scale correctly to any field.
# Values are in the range (-1, 1) relative to each half-axis.
_WAYPOINT_SETS_NORMALISED = [
    # Robot 0 — large figure-8 across the field
    [(-0.67, 0.50), (0.0, 0.0), (0.67, -0.50), (0.0, 0.0)],
    # Robot 1 — diagonal patrol
    [(-0.44, -0.67), (0.44, 0.67)],
    # Robot 2 — wide horizontal sweep
    [(-0.78, 0.17), (0.78, 0.17), (0.78, -0.17), (-0.78, -0.17)],
    # Robot 3 — small loop near centre
    [(0.22, 0.33), (-0.22, 0.33), (-0.22, -0.33), (0.22, -0.33)],
    # Robot 4 — left-half patrol
    [(-0.67, 0.0), (-0.22, 0.67), (-0.22, -0.67)],
    # Robot 5 — right-half patrol
    [(0.67, 0.0), (0.22, 0.67), (0.22, -0.67)],
]

_ARRIVAL_THRESHOLD = 0.15  # metres — how close counts as "reached"


def _scale_waypoints(field_dims: FieldDimensions) -> list[list[Vector2D]]:
    """Return waypoint lists scaled to *field_dims*."""
    L = field_dims.full_field_half_length
    W = field_dims.full_field_half_width
    return [[Vector2D(fx * L, fy * W) for fx, fy in pattern] for pattern in _WAYPOINT_SETS_NORMALISED]


@dataclass
class WanderingMem:
    wp_index: dict[int, int] = field(default_factory=dict)


class WanderingTactic(BaseTactic[WanderingMem]):
    """Every assigned robot continuously patrols a set of waypoints, keyed by
    slot position (robot's position in the sorted outfield roster) rather
    than by robot ID directly, matching the original BT `WanderingStep`'s
    `sorted(friendly_robots)`-indexed waypoint-set assignment."""

    tag = TacticTag.MIXED

    def __init__(self, field_dims: FieldDimensions | None = None):
        self._waypoints = _scale_waypoints(field_dims or STANDARD_FIELD_DIMS)

    def initial_mem(self) -> WanderingMem:
        return WanderingMem()

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: WanderingMem
    ) -> tuple[dict[RobotId, RobotCommand], WanderingMem]:
        commands: dict[RobotId, RobotCommand] = {}
        for slot, robot_id in enumerate(sorted(robot_ids)):
            waypoints = self._waypoints[slot % len(self._waypoints)]

            wp_idx = mem.wp_index.get(robot_id, 0)
            target = waypoints[wp_idx]

            robot = game.friendly_robots[robot_id]
            dist = robot.p.distance_to(target)

            if dist < _ARRIVAL_THRESHOLD:
                wp_idx = (wp_idx + 1) % len(waypoints)
                mem.wp_index[robot_id] = wp_idx
                target = waypoints[wp_idx]

            oren = robot.p.angle_to(target)
            commands[robot_id] = move(game, ctx.motion_controller, robot_id, target, oren)

        return commands, mem


def wandering_strategy(
    outfield_robot_ids: tuple[int, ...], field_dims: FieldDimensions | None = None
) -> AbstractStrategy:
    """Kernel-native replacement for the deleted BT `WanderingStrategy`.

    `goalkeeper_id` stays at its `AbstractStrategy` default (0) even though
    robot 0 is also in the wandering roster — harmless here, same reasoning as
    `tests/motion_planning/_kernel_test_strategies.go_to_point_strategy`
    (every robot in `outfield_robot_ids` always gets a command from the
    tactic, so the goalkeeper tick is always skipped for it).
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        tactic = WanderingTactic(field_dims)
        return KernelSchedulerStrategy(
            tactics={"wander": tactic},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "wander"),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return AbstractStrategy(build_kernel_strategy=_build)


__all__ = ["WanderingTactic", "wandering_strategy"]
