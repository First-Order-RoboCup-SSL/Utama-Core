from entities.game import Game, Robot
from entities.data.vector import Vector2D
from motion_planning.src.motion_controller import MotionController
from config.settings import ROBOT_RADIUS
from entities.data.command import RobotCommand
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from skills.src.utils.move_utils import kick, turn_on_spot
from skills.src.go_to_ball import go_to_ball
from global_utils.math_utils import angle_between_points as _angle_between_points
import numpy as np
import math
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

# Calculates the intersection of 2 rays with the goal
def _shadow(
    start_x: float, start_y: float, angle1: float, angle2: float, goal_x: float
) -> Tuple[float, float]:
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


# Casts a ray along the 2 tangents to each enemy robot, and calls _filter_and_merge_shadows
def _ray_casting(
    point: Vector2D,
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
            if goal_multi * enemy.p.x > goal_multi * point[0]:
                dist: float = math.dist((point[0], point[1]), (enemy.p.x, enemy.p.y))
                angle_to_robot_ = point.angle_to(enemy.p)
                alpha: float = np.arcsin(ROBOT_RADIUS / dist)
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
        # If the interval touches a goal boundary, the best candidate is that boundary.
        if s == goal_y1:
            candidate = s + 0.2 * abs(s + e) / 2
            clearance = e - s  # Full gap length is clearance.
        elif e == goal_y2:
            candidate = e - 0.2 * abs(s + e) / 2
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
    point: Vector2D,
    enemy_robots,
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
) -> float:
    """
    Computes the shot quality based on the open angle to the goal / total angle to the goal.
    Uses the _find_best_shot function to determine the largest open angle.
    """

    # Full angle between the two goalposts
    full_angle = _angle_between_points(
        point, Vector2D(goal_x, goal_y1), Vector2D(goal_x, goal_y2)
    )

    # Use _find_best_shot to get the largest gap
    _, largest_gap = _find_best_shot(
        point, enemy_robots, goal_x, goal_y1, goal_y2
    )

    # Compute the open angle (gap angle)
    open_angle = _angle_between_points(
        point,
        Vector2D(goal_x, largest_gap[0]),
        Vector2D(goal_x, largest_gap[1]),
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


def is_goal_blocked(
    game: Game, best_shot: Tuple[float, float], defenders: List[Robot]
) -> bool:
    """
    Determines whether the goal is blocked by enemy robots (considering them as circles).

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


def score_goal(
    game: Game,
    motion_controller: MotionController,
    shooter_id: int,
    env: SSLStandardEnv = None,
) -> RobotCommand:
    """
    Attempts to score a goal by calculating the best shot angle and executing the kick if possible.
    """
    target_goal_line = game.field.enemy_goal_line
    shooter = game.friendly_robots[shooter_id]
    defender_robots = game.enemy_robots
    ball = game.ball
    goal_x = target_goal_line.coords[0][0]
    goal_y1 = target_goal_line.coords[1][1]
    goal_y2 = target_goal_line.coords[0][1]

    # calculate best shot from the position of the ball
    # TODO: add sampling function to try to find other angles to shoot from that are more optimal
    if defender_robots and shooter and ball:
        best_shot, _ = _find_best_shot(
            ball.p, list(defender_robots.values()), goal_x, goal_y1, goal_y2
        )

        # Safe fall-back if no best shot is found
        if best_shot is None and is_goal_blocked(
            game, (goal_x, goal_y2 - goal_y1), defender_robots
        ):
            return None
        elif best_shot is None and not is_goal_blocked(
            game, (goal_x, goal_y2 - goal_y1), defender_robots
        ):
            best_shot = (goal_y2 + goal_y1) / 2

        shot_orientation = (np.atan2((best_shot - ball.p.y), (goal_x - ball.p.x))) % (
            2 * np.pi
        )

        if env is not None:
            line_points = [
                (shooter.p.x, shooter.p.y),
                (
                    goal_x,
                    shooter.p.y + np.tan(shot_orientation) * (goal_x - shooter.p.x),
                ),
            ]
            env.draw_line(line_points)
            env.draw_point(goal_x, best_shot, color="red")

    if shooter.has_ball:
        logging.debug("robot has ball")
        current_oren = shooter.orientation % (2 * np.pi)
        # if robot has ball and is facing the goal, kick the ball
        # TODO: This should be changed to a smarter metric (ie within the range of tolerance of the shot)
        # Because 0.02 as a threshold is meaningless (different at different distances)
        # TODO: consider also adding a distance from goal threshold
        if abs(current_oren - shot_orientation) * abs(
            goal_x - shooter.p.x
        ) <= 0.05 and not is_goal_blocked(
            game, (goal_x, best_shot), list(defender_robots.values())
        ):
            logger.info("kicking ball")
            robot_command = kick()
        # TODO: Consider also advancing closer to the goal
        else:
            # print("turning on spot to " + str(shot_orientation))
            # print("right now at " + str(current_oren))
            robot_command = turn_on_spot(
                game,
                motion_controller,
                shooter_id,
                shot_orientation,
                dribbling=True,
            )
    else:
        # robot_command = move(game, motion_controller, shooter_id, shooter_id, ball)
        robot_command = go_to_ball(game, motion_controller, shooter_id)
    return robot_command
