from typing import Dict, List, Optional, Tuple

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Ball, Game, Robot
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv

EPS = 1e-5


def align_defenders(
    game: Game,
    defender_parametric_pos: float,
    attacker_position: Vector2D,
    attacker_orientation: Optional[float],
    env: Optional[SSLStandardEnv],
) -> Tuple[float, float]:
    """Calculates the next point on the defense area that the robots should go to defender_position is in terms of t on
    the parametric curve."""

    NO_MOVE_THRES = 1.5

    # Start by getting the current position of the defenders
    defender_pos = calculate_defense_area(game, defender_parametric_pos)

    # logger.debug(f"DEFENDER {dx} {dy}")
    goal_centre_x = game.field.my_goal_line.coords[0][0]

    if attacker_orientation is None or attacker_orientation == 0:
        # In case there is no ball velocity or attackers, use centre of goal
        predicted_goal_position = Vector2D(goal_centre_x, 0)
    else:
        predicted_goal_position = Vector2D(
            goal_centre_x,
            clamp_to_goal_width(predict_goal_y_location(game, attacker_position, attacker_orientation)),
        )

    if env:
        env.draw_line([predicted_goal_position, attacker_position], width=1, color="green")
        env.draw_line([predicted_goal_position, defender_pos], width=1, color="yellow")

        poly = []
        for t in range(round(1000 * np.pi / 2), round(1000 * 3 * np.pi / 2) + 1):
            poly.append(calculate_defense_area(game, clamp_to_parametric(t / 1000)))
        env.draw_polygon(poly, width=3)

    goal_to_defender = defender_pos - predicted_goal_position
    goal_to_attacker = attacker_position - predicted_goal_position

    side = ccw(goal_to_defender, goal_to_attacker)
    angle = ang_between(goal_to_defender, goal_to_attacker)

    if not game.my_team_is_right:
        side *= -1

    if np.rad2deg(angle) > NO_MOVE_THRES:
        # Move to the correct side
        next_t = step_curve(defender_parametric_pos, side)
        # logger.debug(
        #     f"RAW NEXT {calculate_defense_area(game, clamp_to_parametric(next_t))} {side} {angle}"
        # )
        return calculate_defense_area(game, next_t)
    else:
        return calculate_defense_area(game, defender_parametric_pos)


def calculate_defense_area(game: Game, t: float) -> Vector2D:
    """Defenders' path around the goal in the form of a rounded rectangle.

    around the penalty box. Ensures defenders don't go inside the box and stay
    right on the edge. This edge of the penalty box is theirs - attackers are
    not allowed this close to the box.

    x = a * ((1 - r) * |cos(t)| * cos(t) + r * cos(t))
    y = a * ((1 - r) * |sin(t)| * sin(t) + r * sin(t))

    https://www.desmos.com/calculator/nmaf7rpmnw
    """
    MIN_T = np.pi / 2
    MAX_T = 3 * np.pi / 2

    t = max(MIN_T, min(t, MAX_T))

    cos_t = np.cos(t)
    sin_t = np.sin(t)

    a, r = 1.1, 2.1
    rp = Vector2D(
        a * ((1 - r) * (abs(cos_t) * cos_t) + r * cos_t),
        a * ((1 - r) * (abs(sin_t) * sin_t) + r * sin_t),
    )

    # This slows everything down sooo much that everything breaks (ask Fred if still confused)
    # goal_centre_x, _ = game.field.my_goal_line.coords[0]

    if game.my_team_is_right:
        goal_centre_x = 4.5
    else:
        goal_centre_x = -4.5

    return make_relative_to_goal_centre(goal_centre_x, rp)


def make_relative_to_goal_centre(goal_centre_x: float, p: Vector2D) -> Vector2D:
    if goal_centre_x < 0:  # Left goal
        return Vector2D(goal_centre_x - p.x, p.y)
    else:
        return Vector2D(goal_centre_x + p.x, p.y)


def predict_goal_y_location(game: Game, shooter_position: Vector2D, orientation: float) -> float:
    dx, dy = np.cos(orientation), np.sin(orientation)
    gx, _ = game.field.my_goal_line.coords[0]
    if dx == 0:
        return float("inf")
    t = (gx - shooter_position[0]) / dx
    return shooter_position[1] + t * dy


def to_defense_parametric(game: Game, p: Vector2D) -> float:
    """Given a point p on the defenders' parametric curve (as defined by calculate_defense_area), returns the parameter
    value t which would give rise to this point."""

    # Ternary search the paramater, minimising the Euclidean distance between
    # the point corresponding to the predicted t and the actual point. We
    # could potentially use length along the curve (I think you can get this
    # from polar coordinates? But for a semicircle ish curve this works fine)
    lo = np.pi / 2
    hi = 3 * np.pi / 2
    EPS = 1e-6

    while (hi - lo) > EPS:
        mi1 = lo + (hi - lo) / 3
        mi2 = lo + 2 * (hi - lo) / 3

        pred1 = calculate_defense_area(game, mi1)
        pred2 = calculate_defense_area(game, mi2)

        dist1 = p.distance_to(pred1)
        dist2 = p.distance_to(pred2)

        if dist1 < dist2:
            hi = mi2
        else:
            lo = mi1

    t = lo
    return clamp_to_parametric(t)


def find_likely_enemy_shooter(enemy_robots: Dict[int, Robot], ball: Ball) -> List[Robot]:
    unique_shooters: Dict[int, Robot] = {}

    for robot_id, robot in enemy_robots.items():
        if np.linalg.norm(robot.p - ball.p.to_2d()) < 0.2:
            unique_shooters[robot_id] = robot

    return list(unique_shooters.values())


def step_curve(t: float, direction: int):
    STEP_SIZE = 0.0872665 * 2
    if direction == 0:
        return t
    return direction * STEP_SIZE + t


def clamp_to_goal_width(y: float) -> float:
    return max(min(y, 0.5), -0.5)


def clamp_to_parametric(t: float) -> float:
    # parametric is between pi /2 and 3pi / 2
    return min(3 * np.pi / 2, max(t, np.pi / 2))


def ccw(v1: Vector2D, v2: Vector2D) -> int:
    # 1 if v1 is c-cw of v2, -1 if v1 is cw of v2, 0 if colinear
    mag = np.cross(v1, v2)
    if abs(mag) < EPS:
        return 0
    if mag > 0:
        return 1
    else:
        return -1


# Repeated I will remove
def ang_between(v1: Vector2D, v2: Vector2D) -> float:
    res = np.dot(v1, v2) / (v1.mag() * v2.mag())
    if res > 0:
        res -= EPS
    else:
        res += EPS
    assert -1 <= res <= 1, f"{v1} {v2} {res}"

    return np.arccos(res)


def velocity_to_orientation(p: Tuple[float, float]) -> float:
    # Takes a velocity and converts to orientation in radians identical to robot orientation
    res = np.atan2(round(p[1], 3), round(p[0], 3))
    if res < 0:
        res += 2 * np.pi
    return res
