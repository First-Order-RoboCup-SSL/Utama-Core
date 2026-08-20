"""Plain-function geometry/condition helpers shared by pass-and-attack tactics.

Ported from `utama_strategy.functional.skills`, with one change: the
original imported `find_best_shot` from `utama_strategy.utils.score_goal_utils`,
a Strategy-repo-private near-duplicate of Core's own `_find_best_shot`
(same ray-casting/shadow algorithm, forked at some point — see that
module's docstring). Since this code now lives in Core, it uses Core's
own `_find_best_shot` directly instead of carrying the duplicate forward.

Each function is `(game, ...) -> value`: no blackboard, no py_trees
Status, no hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.skills.src.score_goal import (  # noqa: F401  (re-exported)
    _find_best_shot,
    is_goal_blocked,
)

ORIENTATION_TOLERANCE_RAD = 0.05


def has_ball(game: Game, robot_id: int, visual: bool = False, capture_distance: float = 0.15) -> bool:
    """`capture_distance` default: `ROBOT_RADIUS + BALL_RADIUS` (contact distance)
    is ~0.1115m. The previous 0.12m default left only ~0.008m of margin above
    contact — well inside typical per-tick simulator jitter (observed: a
    stationary dribbling robot's distance-to-ball oscillates by ~0.01-0.02m
    tick to tick), so `has_ball(..., visual=True)` chattered True/False every
    tick right at pickup, which made every caller's "if has_ball: X else: Y"
    branch flip every tick too and never make sustained progress in either
    branch. 0.15m gives real margin above contact distance.
    """
    robot = game.friendly_robots[robot_id]
    if visual:
        return robot.p.distance_to(game.ball.p.to_2d()) < capture_distance
    return bool(robot.has_ball)


def at_target(game: Game, robot_id: int, target: Vector2D, tolerance: float = 0.08) -> bool:
    robot = game.friendly_robots[robot_id]
    return robot.p.distance_to(target) <= tolerance


def oriented_towards(
    game: Game, robot_id: int, target_orientation: float, tolerance: float = ORIENTATION_TOLERANCE_RAD
) -> bool:
    robot = game.friendly_robots[robot_id]
    diff = (target_orientation - robot.orientation + math.pi) % (2 * math.pi) - math.pi
    return abs(diff) <= tolerance


def clamp_to_field(position: Vector2D, game: Game, margin: float = 0.3) -> Vector2D:
    field = game.field
    x = max(-field.half_length + margin, min(field.half_length - margin, position.x))
    y = max(-field.half_width + margin, min(field.half_width - margin, position.y))
    return Vector2D(x, y)


def intercept_point(
    game: Game,
    passer_id: int,
    receiver_id: int,
    min_intercept_distance: float = 0.5,
) -> tuple[Vector2D, float]:
    """Returns (intercept_position, intercept_orientation)."""
    passer = game.friendly_robots[passer_id]
    receiver = game.friendly_robots[receiver_id]
    ball_pos = game.ball.p.to_2d()

    trajectory_direction = Vector2D(math.cos(passer.orientation), math.sin(passer.orientation))
    projection_t = (receiver.p - ball_pos).dot(trajectory_direction)
    intercept_distance = max(projection_t, min_intercept_distance)
    intercept_position = clamp_to_field(ball_pos + trajectory_direction * intercept_distance, game)

    intercept_orientation = math.atan2(ball_pos.y - receiver.p.y, ball_pos.x - receiver.p.x)
    return intercept_position, intercept_orientation


def enemy_positions(game: Game) -> list[Vector2D]:
    return [enemy.p for enemy in game.enemy_robots.values() if enemy is not None]


def _distance_to_segment(point: Vector2D, start: Vector2D, end: Vector2D) -> float:
    segment = end - start
    segment_len_sq = segment.dot(segment)
    if segment_len_sq <= 1e-12:
        return point.distance_to(start)
    projection = max(0.0, min(1.0, (point - start).dot(segment) / segment_len_sq))
    closest = start + segment * projection
    return point.distance_to(closest)


def segment_clearance(start: Vector2D, end: Vector2D, obstacles: list[Vector2D]) -> float:
    """Minimum distance from any obstacle to the start->end segment. 10.0 if none."""
    if not obstacles:
        return 10.0
    return min(_distance_to_segment(obstacle, start, end) for obstacle in obstacles)


def segment_blocked(
    start: Vector2D, end: Vector2D, obstacles: list[Vector2D], clearance: float = ROBOT_RADIUS + 0.15
) -> bool:
    return segment_clearance(start, end, obstacles) <= clearance


def enemy_goal_line(game: Game) -> tuple[float, float, float]:
    goal_line = game.field.enemy_goal_line
    goal_x = float(goal_line[0][0])
    goal_y1 = min(float(goal_line[0][1]), float(goal_line[1][1]))
    goal_y2 = max(float(goal_line[0][1]), float(goal_line[1][1]))
    return goal_x, goal_y1, goal_y2


def in_own_defense_area(game: Game, point: Vector2D) -> bool:
    """True if `point` is inside our own defense area.

    Uses `field.my_defense_area` — the same geometry the CustomReferee's
    `DefenseAreaRule` derives from (`half_defense_area_depth`/`width`), so a
    tactic deciding legality by this check agrees with the referee.
    """
    defense_area = game.field.my_defense_area
    front_x = float(defense_area[1][0])
    goal_x = game.field.my_goal_line[0][0]
    half_width = abs(float(defense_area[0][1]))
    x_inside = (point.x - front_x) * (goal_x - front_x) >= 0.0
    return x_inside and abs(point.y) <= half_width


def ball_in_own_defense_area(game: Game) -> bool:
    """True if the ball center is inside our own defense area (see `in_own_defense_area`)."""
    return in_own_defense_area(game, game.ball.p.to_2d())


def clamp_outside_own_defense_area(game: Game, point: Vector2D, margin: float = 2.0 * ROBOT_RADIUS + 0.05) -> Vector2D:
    """Clamp a target point to just outside our own defense area's front edge.

    The `DefenseAreaRule` fouls any outfield robot entering the area (the
    keeper owns the box), so tactics that route robots near their own goal —
    shot-line defenders, carriers chasing a loose ball — must never target
    inside it. Keep-out is enforced on x with one robot-diameter margin;
    the y-coordinate is preserved unchanged (the area only spans
    `half_defense_area_width`, so a clamped x alone is already outside).
    """
    defense_area = game.field.my_defense_area
    front_x = float(defense_area[1][0])
    sign = 1.0 if game.my_team_is_right else -1.0
    exit_x = front_x - sign * margin
    if sign > 0 and point.x > exit_x:
        return Vector2D(exit_x, point.y)
    if sign < 0 and point.x < exit_x:
        return Vector2D(exit_x, point.y)
    return point


def own_defense_area_exit_point(game: Game, at_y: float, margin: float = 2.0 * ROBOT_RADIUS + 0.05) -> Vector2D:
    """A hold point just outside our own defense area's front edge at `at_y`.

    For a defender/carrier that needs to stand near a ball that is inside
    our own area (which outfield robots may not enter): hold the edge
    closest to the ball, with y clamped inside the area's width so the
    point is the nearest legal standing spot.
    """
    defense_area = game.field.my_defense_area
    front_x = float(defense_area[1][0])
    half_width = abs(float(defense_area[0][1]))
    sign = 1.0 if game.my_team_is_right else -1.0
    exit_x = front_x - sign * margin
    y = max(-(half_width - margin), min(half_width - margin, at_y))
    return Vector2D(exit_x, y)


def clamp_outside_enemy_defense_area(
    game: Game, point: Vector2D, margin: float = 2.0 * ROBOT_RADIUS + 0.05
) -> Vector2D:
    """Clamp a target point to just outside the enemy's defense area front edge.

    `DefenseAreaRule` fouls attacker encroachment into the enemy box too (not
    just our own — `attacker_infringement=True` is the referee default), so
    any attacking tactic that scripts a target close to the enemy goal
    (overload/relay finishing runs, decoy lures, switch-of-play runners) must
    clamp it the same way `clamp_outside_own_defense_area` clamps defensive
    targets. Mirror image of that function: the enemy goal is on the
    opposite side from ours, so the clamp direction is `-sign` instead of
    `sign`.
    """
    defense_area = game.field.enemy_defense_area
    front_x = float(defense_area[1][0])
    sign = -1.0 if game.my_team_is_right else 1.0
    exit_x = front_x - sign * margin
    if sign > 0 and point.x > exit_x:
        return Vector2D(exit_x, point.y)
    if sign < 0 and point.x < exit_x:
        return Vector2D(exit_x, point.y)
    return point


def find_best_shot(
    point: Vector2D, enemy_robots: list, goal_x: float, goal_y1: float, goal_y2: float
) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """Thin passthrough to Core's own shadow/ray-casting shot finder."""
    return _find_best_shot(point, enemy_robots, goal_x, goal_y1, goal_y2)


@dataclass(frozen=True)
class PassSetupScore:
    passer_position: Vector2D
    receiver_position: Vector2D
    score: float


def score_pass_setup(
    game: Game,
    passer_position: Vector2D,
    receiver_position: Vector2D,
    min_pass_distance: float = 0.7,
    min_robot_clearance: float = ROBOT_RADIUS + 0.15 + 0.2,
) -> Optional[PassSetupScore]:
    """Score a candidate (passer_position, receiver_position) setup pair.

    Returns None if the pair is infeasible (too close together, blocked
    pass, no open shot, blocked shot, or too close to a friendly robot).
    """
    pass_distance = passer_position.distance_to(receiver_position)
    if pass_distance < min_pass_distance:
        return None

    friendly_positions = [robot.p for robot in game.friendly_robots.values() if robot is not None]
    for robot_pos in friendly_positions:
        if passer_position.distance_to(robot_pos) < min_robot_clearance:
            return None
        if receiver_position.distance_to(robot_pos) < min_robot_clearance:
            return None

    enemies = enemy_positions(game)
    if segment_blocked(passer_position, receiver_position, enemies):
        return None

    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    best_shot_y, largest_gap = find_best_shot(
        receiver_position, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2
    )
    if best_shot_y is None or largest_gap is None:
        return None

    shot_target = Vector2D(goal_x, best_shot_y)
    if segment_blocked(receiver_position, shot_target, enemies):
        return None

    pass_clearance = segment_clearance(passer_position, receiver_position, enemies)
    shot_gap = largest_gap[1] - largest_gap[0]
    distance_to_goal_ratio = abs(receiver_position.x - goal_x) / max(2.0 * abs(goal_x), 1e-6)
    score = shot_gap + 0.3 * pass_clearance - 0.2 * distance_to_goal_ratio - 0.03 * pass_distance

    return PassSetupScore(passer_position=passer_position, receiver_position=receiver_position, score=score)
