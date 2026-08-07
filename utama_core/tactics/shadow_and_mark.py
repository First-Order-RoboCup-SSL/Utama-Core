"""Shadow-and-mark defense tactic — shot-line shadowing plus opponent marking.

New tactical logic for defenders beyond the first two, not ported from
Utama-Strategy. `defend_parameter` (already in Core, reused here as-is)
already handles shot-line shadowing for 1-2 defenders, including its own
dynamic side-selection for exactly 2 — see `tactics/defense.py`, which wraps
that function directly and is left unchanged for the 1-2 robot case. That
function has no concept of a 3rd, 4th, or 5th defender, so it cannot simply
be called per-robot for larger defensive groups: nothing stops it from
sending every robot to the same post. This tactic makes the missing decision
explicit — the first two assigned robots shadow via `defend_parameter`
exactly like `DefenseTactic`; any further robots each mark the nearest
not-yet-marked enemy robot (denying that opponent space/a passing lane)
instead of clustering on the same shot line.

Reuses `defend_parameter`, `go_to_point`, and `proximity_lookup`/plain
distance comparisons — all existing motion/geometry primitives, not
tactics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import BaseTactic, RobotId
from utama_core.skills.src.defend_parameter import defend_parameter
from utama_core.skills.src.go_to_point import go_to_point

_MARK_STANDOFF = 0.6  # metres — mark from this distance on the goal side of the opponent, not on top of them


def _mark_target(game: Game, opponent_id: int) -> Vector2D:
    goal_x = game.field.my_goal_line[0][0]
    opponent = game.enemy_robots[opponent_id]
    direction = 1.0 if goal_x < opponent.p.x else -1.0
    return Vector2D(opponent.p.x + direction * _MARK_STANDOFF, opponent.p.y)


def _assign_marks(game: Game, marker_ids: tuple[int, ...]) -> dict[int, int]:
    """Nearest-unmarked-opponent-first greedy assignment, marker by marker."""
    enemy_ids = list(game.enemy_robots.keys())
    assignment: dict[int, int] = {}
    remaining_enemies = set(enemy_ids)

    for marker_id in marker_ids:
        if not remaining_enemies:
            break
        marker_pos = game.friendly_robots[marker_id].p
        closest_enemy = min(remaining_enemies, key=lambda eid: marker_pos.distance_to(game.enemy_robots[eid].p))
        assignment[marker_id] = closest_enemy
        remaining_enemies.discard(closest_enemy)

    return assignment


@dataclass
class ShadowAndMarkMem:
    """No cross-tick state: shadow assignment and marks are recomputed every tick."""


class ShadowAndMarkTactic(BaseTactic[ShadowAndMarkMem]):
    """Up to 5 defenders: first two shadow the shot line, the rest mark opponents.

    Marking assignment is greedy nearest-first per tick, not sticky — a
    marker can switch targets tick to tick if a closer unmarked opponent
    appears. Acceptable for now since marking has no phase/commitment state
    to disrupt (unlike `LeadAndSupportTactic`'s leader role); revisit only if
    mark-flapping is observed to be a real problem.
    """

    def make_initial_mem(self) -> ShadowAndMarkMem:
        return ShadowAndMarkMem()

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: ShadowAndMarkMem
    ) -> tuple[dict[RobotId, RobotCommand], ShadowAndMarkMem]:
        shadow_ids = robot_ids[:2]
        marker_ids = robot_ids[2:]

        commands: dict[RobotId, RobotCommand] = {
            robot_id: defend_parameter(game, ctx.motion_controller, robot_id, env=ctx.rsim_env)
            for robot_id in shadow_ids
        }

        marks = _assign_marks(game, marker_ids)
        for marker_id in marker_ids:
            opponent_id = marks.get(marker_id)
            if opponent_id is None:
                # More markers than opponents to mark — hold position via a
                # no-op shadow assignment rather than leaving no command.
                commands[marker_id] = defend_parameter(game, ctx.motion_controller, marker_id, env=ctx.rsim_env)
                continue
            target = _mark_target(game, opponent_id)
            commands[marker_id] = go_to_point(
                game=game, motion_controller=ctx.motion_controller, robot_id=marker_id, target_coords=target
            )

        return commands, mem
