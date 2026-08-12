"""Lead-and-support attack tactic — one ball carrier, N-1 supporting robots.

New tactical logic, not ported from Utama-Strategy's `plays/`/`strategies/`
(the user explicitly asked for genuinely new tactics rather than a renamed
2v1/go-to-space). `two_robot_attack`'s fixed passer/receiver pattern doesn't
generalize past two robots without deciding what the other robots do — this
tactic makes that decision: closest-to-ball leads (dribble toward goal, shoot
when a lane opens, or pass to the best-scoring support point), every other
assigned robot continuously re-picks the best open support point on the
attacking half. Reuses existing skill/geometry primitives
(`go_to_ball`/`go_to_point`/`move`, and `shared/pass_and_score_geometry`'s
shot-finding and pass-scoring functions) — the primitives are fundamental
motion/geometry, not tactical decisions, so they were kept rather than
reinvented.

Robot-count-agnostic per the design doc's §7 stance (no scheduler-level
splitting policy yet): works with 1 robot (leader only, no support) up to
however many are assigned. Leader role is re-evaluated every tick except
while `is_committed()` — see below — mirroring `two_robot_attack`'s setup-phase
gating, generalized: once the leader has the ball and is not merely passing
setup, re-picking "closest to ball" every tick is wrong for the same reason
it was wrong there (the ball is transiently closer to a receiver than the
carrier during a pass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.object import TeamType
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import BaseTactic, RobotId
from utama_core.shared.pass_and_score_geometry import (
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    oriented_towards,
    segment_blocked,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import empty_command, kick, turn_on_spot

_MIN_SUPPORT_SEPARATION = 0.9  # metres — supports must not crowd each other or the leader
_SUPPORT_FORWARD_BIAS = 0.35  # weight favouring support points closer to the enemy goal


def _candidate_support_points(game: Game, count: int) -> list[Vector2D]:
    """Grid of candidate points across the attacking half, in front of the ball."""
    half_length = game.field.half_length
    half_width = game.field.half_width
    goal_x, _, _ = enemy_goal_line(game)
    ball_x = game.ball.p.to_2d().x

    x_lo = max(min(ball_x, goal_x), -half_length + 0.5)
    x_hi = min(max(ball_x, goal_x), half_length - 0.5)
    if x_hi <= x_lo:
        x_lo, x_hi = -half_length + 0.5, half_length - 0.5

    cols = max(2, int(count**0.5) + 2)
    rows = max(2, count)
    points = []
    for i in range(cols):
        x = x_lo + (x_hi - x_lo) * (i + 0.5) / cols
        for j in range(rows):
            y = -half_width + 0.6 + (2 * half_width - 1.2) * (j + 0.5) / rows
            points.append(Vector2D(x, y))
    return points


def _score_support_point(game: Game, point: Vector2D, leader_pos: Vector2D, taken: list[Vector2D]) -> Optional[float]:
    for other in taken:
        if point.distance_to(other) < _MIN_SUPPORT_SEPARATION:
            return None
    if point.distance_to(leader_pos) < _MIN_SUPPORT_SEPARATION:
        return None

    enemies = enemy_positions(game)
    if segment_blocked(leader_pos, point, enemies, clearance=0.3):
        lane_open = 0.0
    else:
        lane_open = 1.0

    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    best_shot_y, gap = find_best_shot(point, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)
    shot_openness = (gap[1] - gap[0]) if (best_shot_y is not None and gap is not None) else 0.0

    forward_progress = abs(point.x - goal_x) * -1.0  # closer to goal_x is better
    return 2.0 * lane_open + shot_openness + _SUPPORT_FORWARD_BIAS * forward_progress


def _pick_support_positions(game: Game, leader_pos: Vector2D, support_count: int) -> list[Vector2D]:
    if support_count <= 0:
        return []
    candidates = _candidate_support_points(game, support_count)
    chosen: list[Vector2D] = []
    for _ in range(support_count):
        best_point, best_score = None, None
        for point in candidates:
            score = _score_support_point(game, point, leader_pos, chosen)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_point, best_score = point, score
        if best_point is None:
            # Every candidate conflicted (crowded field) — fall back to the
            # single least-bad point ignoring separation from other supports.
            best_point = min(candidates, key=lambda p: p.distance_to(leader_pos))
        chosen.append(best_point)
    return chosen


def choose_leader(game: Game, robot_ids: tuple[int, ...]) -> int:
    """Closest-to-ball becomes leader. Falls back to `robot_ids[0]`."""
    closest, _distance = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    if closest is not None and closest.id in robot_ids:
        return closest.id
    return robot_ids[0]


@dataclass
class LeadAndSupportMem:
    leader_id: Optional[int] = None


class LeadAndSupportTactic(BaseTactic[LeadAndSupportMem]):
    """One ball-carrying leader plus N-1 supporting robots holding open space.

    The leader dribbles toward goal and shoots as soon as a shot lane opens;
    otherwise it holds the ball while supports resettle. There is no
    passing/receiving handoff sequencing here (unlike `two_robot_attack`) —
    supports exist to occupy space and pull opponents out of position, not
    to be passed to. `is_committed()` covers only "leader has the ball and is
    lined up to shoot," a short-lived window, so this tactic is rarely a
    long-term blocker on reassignment.
    """

    def initial_mem(self) -> LeadAndSupportMem:
        return LeadAndSupportMem()

    def is_committed(self, game: Game, mem: LeadAndSupportMem) -> bool:
        if mem.leader_id is None:
            return False
        return has_ball(game, mem.leader_id)

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: LeadAndSupportMem
    ) -> tuple[dict[RobotId, RobotCommand], LeadAndSupportMem]:
        if mem.leader_id is None or mem.leader_id not in robot_ids or not self.is_committed(game, mem):
            mem.leader_id = choose_leader(game, robot_ids)

        leader_id = mem.leader_id
        support_ids = tuple(rid for rid in robot_ids if rid != leader_id)

        commands: dict[RobotId, RobotCommand] = {}
        leader_pos = game.friendly_robots[leader_id].p

        if not has_ball(game, leader_id):
            commands[leader_id] = go_to_ball(game=game, motion_controller=ctx.motion_controller, robot_id=leader_id)
        else:
            goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
            best_shot_y, gap = find_best_shot(leader_pos, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)
            if best_shot_y is None:
                commands[leader_id] = empty_command(dribbler_on=True)
            else:
                target_oren = leader_pos.angle_to(Vector2D(goal_x, best_shot_y))
                if oriented_towards(game, leader_id, target_oren):
                    commands[leader_id] = kick()
                else:
                    commands[leader_id] = turn_on_spot(
                        game=game,
                        motion_controller=ctx.motion_controller,
                        robot_id=leader_id,
                        target_oren=target_oren,
                        dribbling=True,
                    )

        support_points = _pick_support_positions(game, leader_pos, len(support_ids))
        for robot_id, point in zip(support_ids, support_points):
            commands[robot_id] = go_to_point(
                game=game, motion_controller=ctx.motion_controller, robot_id=robot_id, target_coords=point
            )

        return commands, mem
