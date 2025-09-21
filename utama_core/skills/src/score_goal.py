import logging
import math
from typing import List, Tuple

import numpy as np

from utama_core.config.settings import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game, Robot
from utama_core.global_utils.math_utils import (
    angle_between_points as _angle_between_points,
)
from utama_core.motion_planning.src.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.utils.move_utils import kick, turn_on_spot

logger = logging.getLogger(__name__)


# Calculates the intersection of 2 rays with the goal
def _shadow(start_x: float, start_y: float, angle1: float, angle2: float, goal_x: float) -> Tuple[float, float]:
    slope1: float = np.tan(angle1)
    slope2: float = np.tan(angle2)
    shadow_start: float = start_y + slope1 * (goal_x - start_x)
    shadow_end: float = start_y + slope2 * (goal_x - start_x)
    return tuple(sorted((shadow_start, shadow_end)))


# Filters the shadows to only keep the ones relevant to the shot and merges overlapping shadows
def _filter_and_merge_shadows(
    shadows: List[Tuple[float, float]], goal_y1: float, goal_y2: float
) -> List[Tuple[float, float]]:
    valid_shadows: List[Tuple[float, float]] = []

    for start, end in shadows:
        # Ensure start is less than end
        if start > end:
            start, end = end, start

        clipped_start = max(start, goal_y1)
        clipped_end = min(end, goal_y2)
        if clipped_start < clipped_end:
            valid_shadows.append((clipped_start, clipped_end))

    valid_shadows.sort()

    merged_shadows: List[Tuple[float, float]] = []
    for start, end in valid_shadows:
        if not merged_shadows or merged_shadows[-1][1] < start:
            merged_shadows.append((start, end))
        else:
            merged_shadows[-1] = (
                merged_shadows[-1][0],
                max(merged_shadows[-1][1], end),
            )

    return merged_shadows


# Casts a ray along the 2 tangents to each enemy robot, and calls _filter_and_merge_shadows
def _ray_casting(
    point: Vector2D,
    enemy_robots: List[Robot],
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
) -> List[Tuple[float, float]]:
    shadows: List[Tuple[float, float]] = []

    goal_multi = -1 if goal_x < 0 else 1  # flips the goalward direction if we are shooting left
    for enemy in enemy_robots:
        if enemy:
            if goal_multi * enemy.p.x > goal_multi * point[0]:
                dist: float = point.distance_to(enemy.p)
                angle_to_robot_ = point.angle_to(enemy.p)

                if dist <= ROBOT_RADIUS:
                    alpha: float = math.pi / 2  # treat overlaps as fully blocking
                else:
                    ratio = ROBOT_RADIUS / dist
                    alpha = np.arcsin(np.clip(ratio, -1.0, 1.0))
                shadows.append(
                    _shadow(
                        point[0],
                        point[1],
                        angle_to_robot_ + alpha,
                        angle_to_robot_ - alpha,
                        goal_x,
                    )
                )
                shadows = _filter_and_merge_shadows(shadows, goal_y1, goal_y2)
    return shadows


# Finds the biggest area of the goal that doesn't have a shadow (the biggest gap) and finds its midpoint for best shot
# TODO: could add heuristics to prefer shots closer to the goalpost
def _find_best_shot(
    point: Vector2D,
    enemy_robots: List[Vector2D],
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
) -> Tuple[float, Vector2D]:
    """Determines the best y-coordinate along the goal line (at x = goal_x) to shoot, such that the shot is farthest
    from any enemy robots' shadows.

    Args:
        point: The (x, y) coordinate from which the shot is being taken.
        enemy_robots: List of (x, y) positions for enemy robots.
        goal_x: The x-coordinate of the goal line.
        goal_y1: The smallest y-coordinate of the goal.
        goal_y2: The largest y-coordinate of the goal.

    Returns:
        A tuple containing:
          - best_shot: the y-coordinate of the optimal shot.
          - largest_gap: a tuple (start, end) representing the open interval where the shot lies.
        If there is no open space, returns (None, None).
    """
    # Get shadow intervals from enemy robots along the goal line.
    shadows = _ray_casting(
        point,
        enemy_robots=enemy_robots,
        goal_x=goal_x,
        goal_y1=goal_y1,
        goal_y2=goal_y2,
    )

    # If no shadows exist, the entire goal is open.
    if not shadows:
        best_shot = (goal_y1 + goal_y2) / 2
        return best_shot, (goal_y1, goal_y2)

    # Sort the shadow intervals by their starting y-coordinate.
    shadows.sort(key=lambda interval: interval[0])

    # Merge overlapping or adjacent shadow intervals.
    merged_shadows = [shadows[0]]
    for current in shadows[1:]:
        last = merged_shadows[-1]
        if current[0] <= last[1]:
            # Overlapping or adjacent intervals; merge them.
            merged_shadows[-1] = (last[0], max(last[1], current[1]))
        else:
            merged_shadows.append(current)

    # Find open spaces on the goal line between merged shadows.
    open_spaces: List[Tuple[float, float]] = []

    # Check for an open interval before the first shadow.
    if merged_shadows[0][0] > goal_y1:
        open_spaces.append((goal_y1, merged_shadows[0][0]))

    # Check for gaps between consecutive shadows.
    for i in range(1, len(merged_shadows)):
        prev_shadow = merged_shadows[i - 1]
        curr_shadow = merged_shadows[i]
        if curr_shadow[0] > prev_shadow[1]:
            open_spaces.append((prev_shadow[1], curr_shadow[0]))

    # Check for an open interval after the last shadow.
    if merged_shadows[-1][1] < goal_y2:
        open_spaces.append((merged_shadows[-1][1], goal_y2))

    # If there are no open intervals, no shot is possible.
    if not open_spaces:
        return None, None

    # Evaluate each open interval: choose a candidate shot and compute its "clearance"
    # (i.e. the distance to the nearest shadow boundary).
    best_candidate = None
    best_gap = None
    best_clearance = -1
    for interval in open_spaces:
        s, e = interval
        gap_length = e - s
        # If the interval touches a goal boundary, the best candidate is that boundary.
        is_lower_bound = np.isclose(s, goal_y1, rel_tol=0.0, abs_tol=1e-6)
        is_upper_bound = np.isclose(e, goal_y2, rel_tol=0.0, abs_tol=1e-6)

        if is_lower_bound:
            candidate = s + 0.3 * gap_length
            clearance = gap_length
        elif is_upper_bound:
            candidate = e - 0.3 * gap_length
            clearance = gap_length
        else:
            candidate = (s + e) / 2
            clearance = gap_length / 2

        if clearance > best_clearance:
            best_clearance = clearance
            best_candidate = candidate
            best_gap = interval

    return best_candidate, best_gap


def find_shot_quality(
    point: Vector2D,
    enemy_robots,
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
) -> float:
    """Computes the shot quality based on the open angle to the goal / total angle to the goal.

    Uses the _find_best_shot function to determine the largest open angle.
    """

    # Full angle between the two goalposts
    full_angle = _angle_between_points(point, Vector2D(goal_x, goal_y1), Vector2D(goal_x, goal_y2))

    # Use _find_best_shot to get the largest gap
    best_target, largest_gap = _find_best_shot(point, enemy_robots, goal_x, goal_y1, goal_y2)

    if not largest_gap or best_target is None:
        return 0.0

    gap_start, gap_end = largest_gap
    if gap_end - gap_start <= 0:
        return 0.0

    # Compute the open angle (gap angle)
    open_angle = _angle_between_points(
        point,
        Vector2D(goal_x, gap_start),
        Vector2D(goal_x, gap_end),
    )

    distance_to_goal_ratio = (np.absolute(point.x - goal_x)) / np.absolute(2 * goal_x)

    distance_to_goal_weight = 0.4

    # Normalize shot quality
    shot_quality = open_angle / full_angle - distance_to_goal_weight * distance_to_goal_ratio if full_angle > 0 else 0.0

    # protect against negative shot quality
    return max(0.0, shot_quality)


def is_goal_blocked(game: Game, best_shot: Tuple[float, float], defenders: List[Robot]) -> bool:
    """Determines whether the goal is blocked by enemy robots (considering them as circles).

    :param game: The game state containing robot and ball positions.
    :return: True if the goal is blocked, False otherwise.
    """

    ball_x, ball_y = game.ball.p.x, game.ball.p.y

    # Define the shooting line from ball position in the shooter's direction
    line_start = np.array([ball_x, ball_y])
    line_end = np.array([best_shot[0], best_shot[1]])  # Use the best shot position

    # Helper function: shortest distance from a point to a line segment
    def distance_point_to_line(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        proj = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        proj = np.clip(proj, 0, 1)  # Keep projection within the segment
        closest_point = line_start + proj * line_vec
        return np.linalg.norm(point - closest_point)

    # Check if any enemy robot blocks the shooting path
    for defender in defenders:
        if defender:
            robot_pos = np.array([defender.p.x, defender.p.y])
            distance = distance_point_to_line(robot_pos, line_start, line_end)

            if distance <= ROBOT_RADIUS:  # Consider robot as a circle
                return True  # Shot is blocked

    return False  # No robot is blocking
