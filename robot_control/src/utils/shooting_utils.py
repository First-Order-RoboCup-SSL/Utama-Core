from typing import List, Tuple
import numpy as np
from team_controller.src.config.settings import ROBOT_RADIUS


def point_to_robot_dist(
    point_x: float, point_y: float, robot_x: float, robot_y: float
) -> float:
    return np.sqrt((point_y - robot_y) ** 2 + (point_x - robot_x) ** 2)


def angle_to_robot(
    point_x: float, point_y: float, robot_x: float, robot_y: float
) -> float:
    return np.arctan((robot_y - point_y) / (robot_x - point_x))


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
    enemy_robots: List[Tuple[float, float]],
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
    shoot_in_left_goal: bool,
) -> List[Tuple[float, float]]:
    shadows: List[Tuple[float, float]] = []

    goal_mult = (
        -1 if shoot_in_left_goal else 1
    )  # flips the goalward direction if we are shooting left
    for enemy in enemy_robots:
        if enemy is not None:
            if goal_mult * enemy.x > goal_mult * point.x:
                dist: float = point_to_robot_dist(point.x, point.y, enemy.x, enemy.y)
                angle_to_robot_: float = angle_to_robot(
                    point.x, point.y, enemy.x, enemy.y
                )
                alpha: float = np.arcsin(ROBOT_RADIUS / dist)
                shadows.append(
                    shadow(
                        point.x,
                        point.y,
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
    shoot_in_left_goal: bool,
) -> Tuple[float, Tuple[float, float]]:
    shadows = ray_casting(
        point,
        enemy_robots=enemy_robots,
        goal_x=goal_x,
        goal_y1=goal_y1,
        goal_y2=goal_y2,
        shoot_in_left_goal=shoot_in_left_goal,
    )
    if not shadows:
        return (goal_y2 + goal_y1) / 2

    open_spaces: List[Tuple[float, float]] = []

    if shadows[0][0] > goal_y1:
        open_spaces.append((goal_y1, shadows[0][0]))

    for i in range(1, len(shadows)):
        if shadows[i][0] > shadows[i - 1][1]:
            open_spaces.append((shadows[i - 1][1], shadows[i][0]))

    if shadows[-1][1] < goal_y2:
        open_spaces.append((shadows[-1][1], goal_y2))

    largest_gap: Tuple[float, float] = max(
        open_spaces + [(0, 0)], key=lambda x: x[1] - x[0]
    )
    best_shot: float = (largest_gap[0] + largest_gap[1]) / 2

    return best_shot, largest_gap


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
    full_angle = np.absolute(
        np.arctan2(goal_y2 - point.y, goal_x - point.x)
        - np.arctan2(goal_y1 - point.y, goal_x - point.x)
    )

    # Use find_best_shot to get the largest gap
    _, largest_gap = find_best_shot(
        point, enemy_robots, goal_x, goal_y1, goal_y2, shoot_in_left_goal
    )

    # Compute the open angle (gap angle)
    open_angle = np.absolute(
        np.arctan2(largest_gap[1] - point.y, largest_gap[0] - point.x)
        - np.arctan2(largest_gap[1] - point.y, largest_gap[0] - point.x)
    )

    # Normalize shot quality
    shot_quality = open_angle / full_angle if full_angle > 0 else 0
    return shot_quality
