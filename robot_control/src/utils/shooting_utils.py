from typing import List, Tuple
import numpy as np
import math
from team_controller.src.config.settings import ROBOT_RADIUS
from robot_control.src.utils.pass_quality_utils import PointOnField
from entities.game.game import Game
from entities.game.robot import Robot

# TODO: may change to bounded form from [-pi, pi]
def angle_to_robot(
    point_x: float, point_y: float, robot_x: float, robot_y: float
) -> float:
    return np.arctan((robot_y - point_y) / (robot_x - point_x))

def angle_between_points(main_point, point1, point2):
    """
    Computes the angle (in radians) between two lines originating from main_point
    and passing through point1 and point2.

    Parameters:
    main_point (tuple): The common point (x, y).
    point1 (tuple): First point (x, y).
    point2 (tuple): Second point (x, y).

    Returns:
    float: Angle in degrees between the two lines.
    """
    v1 = np.array([point1.x - main_point.x, point1.y - main_point.y])
    v2 = np.array([point2.x - main_point.x, point2.y - main_point.y])

    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm_product == 0:
        return 0  # Avoid division by zero if vectors are degenerate

    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return angle_rad

# Calculates the intersection of 2 rays with the goal
def shadow(
    start_x: float, start_y: float, angle1: float, angle2: float, goal_x: float
) -> Tuple[float, float]:
    slope1: float = np.tan(angle1)
    slope2: float = np.tan(angle2)
    shadow_start: float = start_y + slope1 * (goal_x - start_x)
    shadow_end: float = start_y + slope2 * (goal_x - start_x)
    return tuple(sorted((shadow_start, shadow_end)))

# Filters the shadows to only keep the ones relevant to the shot and merges overlapping shadows
def filter_and_merge_shadows(
    shadows: List[Tuple[float, float]], goal_y1: float, goal_y2: float
) -> List[Tuple[float, float]]:
    valid_shadows: List[Tuple[float, float]] = []

    for start, end in shadows:
        if start > goal_y1 and start < goal_y2:
            if end > goal_y2:
                end = goal_y2
            valid_shadows.append((start, end))
        elif end > goal_y1 and end < goal_y2:
            if start < goal_y1:
                start = goal_y1
            valid_shadows.append((start, end))

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

# Casts a ray along the 2 tangents to each enemy robot, and calls filter_and_merge_shadows
def ray_casting(
    point: Tuple[float, float],
    enemy_robots: List[Robot],
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
) -> List[Tuple[float, float]]:
    shadows: List[Tuple[float, float]] = []

    goal_multi = (
        -1 if goal_x < 0 else 1
    )  # flips the goalward direction if we are shooting left
    for enemy in enemy_robots:
        if enemy:
            if goal_multi * enemy.x > goal_multi * point[0]:
                dist: float = math.dist((point[0], point[1]), (enemy.x, enemy.y))
                angle_to_robot_: float = angle_to_robot(
                    point[0], point[1], enemy.x, enemy.y
                )
                alpha: float = np.arcsin(ROBOT_RADIUS / dist)
                shadows.append(
                    shadow(
                        point[0],
                        point[1],
                        angle_to_robot_ + alpha,
                        angle_to_robot_ - alpha,
                        goal_x,
                    )
                )
                shadows = filter_and_merge_shadows(shadows, goal_y1, goal_y2)
    return shadows

# Finds the biggest area of the goal that doesn't have a shadow (the biggest gap) and finds its midpoint for best shot
# TODO: could add heuristics to prefer shots closer to the goalpost
def find_best_shot(
    point: Tuple[float, float],
    enemy_robots: List[Tuple[float, float]],
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
) -> Tuple[float, Tuple[float, float]]:
    """
    Determines the best y-coordinate along the goal line (at x = goal_x) to shoot,
    such that the shot is farthest from any enemy robots' shadows.

    Args:
        point: The (x, y) coordinate from which the shot is being taken.
        enemy_robots: List of (x, y) positions for enemy robots.
        goal_x: The x-coordinate of the goal line.
        goal_y1: The lower y-coordinate of the goal.
        goal_y2: The upper y-coordinate of the goal.

    Returns:
        A tuple containing:
          - best_shot: the y-coordinate of the optimal shot.
          - largest_gap: a tuple (start, end) representing the open interval where the shot lies.
        If there is no open space, returns (None, None).
    """
    # Get shadow intervals from enemy robots along the goal line.
    shadows = ray_casting(
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
        # If the interval touches a goal boundary, the best candidate is that boundary.
        if s == goal_y1:
            candidate = s
            clearance = e - s  # Full gap length is clearance.
        elif e == goal_y2:
            candidate = e
            clearance = e - s
        else:
            candidate = (s + e) / 2
            clearance = (e - s) / 2
        
        if clearance > best_clearance:
            best_clearance = clearance
            best_candidate = candidate
            best_gap = interval
            
    return best_candidate, best_gap

def find_shot_quality(
    point: Tuple[float, float],
    enemy_robots,
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
    shoot_in_left_goal,
) -> float:
    """
    Computes the shot quality based on the open angle to the goal / total angle to the goal.
    Uses the find_best_shot function to determine the largest open angle.
    """

    # Full angle between the two goalposts
    full_angle = angle_between_points(
        point, PointOnField(goal_x, goal_y1), PointOnField(goal_x, goal_y2)
    )

    # Use find_best_shot to get the largest gap
    _, largest_gap = find_best_shot(
        point, enemy_robots, goal_x, goal_y1, goal_y2, shoot_in_left_goal
    )

    # Compute the open angle (gap angle)
    open_angle = angle_between_points(
        point,
        PointOnField(goal_x, largest_gap[0]),
        PointOnField(goal_x, largest_gap[1]),
    )

    distance_to_goal_ratio = (np.absolute(point.x - goal_x)) / np.absolute(2 * goal_x)

    distance_to_goal_weight = 0.4

    # Normalize shot quality
    shot_quality = (
        open_angle / full_angle - distance_to_goal_weight * distance_to_goal_ratio
        if full_angle > 0
        else 0
    )
    return shot_quality

def is_goal_blocked(game: Game, best_shot: Tuple[float, float], defenders: List[Robot]) -> bool:
    """
    Determines whether the goal is blocked by enemy robots (considering them as circles).
    
    :param game: The game state containing robot and ball positions.
    :return: True if the goal is blocked, False otherwise.
    """

    ball_x, ball_y = game.ball.x, game.ball.y
    
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

    robot_radius = ROBOT_RADIUS  # Assume field provides robot radius info

    # Check if any enemy robot blocks the shooting path
    for defender in defenders:
        if defender:
            robot_pos = np.array([defender.x, defender.y])
            distance = distance_point_to_line(robot_pos, line_start, line_end)
                        
            if distance <= robot_radius:  # Consider robot as a circle
                return True  # Shot is blocked

    return False  # No robot is blocking
