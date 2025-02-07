import math
import numpy as np
from typing import List, Optional, Tuple

from entities.data.command import RobotCommand
from entities.data.vision import BallData, RobotData

from entities.game.game import Game
from motion_planning.src.pid import PID
import logging

from math import atan2, dist, sqrt, cos, sin, pi, acos, degrees
from global_utils.math_utils import normalise_heading
from motion_planning.src.pid.pid import TwoDPID
from robot_control.src.utils.motion_planning_utils import calculate_robot_velocities

from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from team_controller.src.config.settings import ROBOT_RADIUS

logger = logging.getLogger(__name__)


def empty_command(dribbler_on: bool = False) -> RobotCommand:
    return RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick=0,
        chip=0,
        dribble=dribbler_on,
    )


def kick_ball() -> RobotCommand:
    return RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick=1,
        chip=0,
        dribble=0,
    )


def go_to_ball(
    pid_oren: PID,
    pid_trans: TwoDPID,
    this_robot_data: RobotData,
    robot_id: int,
    ball_data: BallData,
) -> RobotCommand:
    # TODO: add a optional target_oren flag
    target_oren = np.arctan2(
        ball_data.y - this_robot_data.y, ball_data.x - this_robot_data.x
    )
    return calculate_robot_velocities(
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        this_robot_data=this_robot_data,
        robot_id=robot_id,
        target_coords=ball_data,
        target_oren=target_oren,
    )


def face_ball(current: Tuple[float, float], ball: Tuple[float, float]) -> float:
    return np.arctan2(ball[1] - current[1], ball[0] - current[0])


def go_to_point(
    pid_oren: PID,
    pid_trans: PID,
    this_robot_data: RobotData,
    robot_id: int,
    target_coords: Tuple[float, float],
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    return calculate_robot_velocities(
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        this_robot_data=this_robot_data,
        robot_id=robot_id,
        target_coords=target_coords,
        target_oren=target_oren,
        dribbling=dribbling,
    )


def turn_on_spot(
    pid_oren: PID,
    pid_trans: PID,
    this_robot_data: RobotData,
    robot_id: int,
    target_oren: float,
    dribbling: bool = False,
    pivot_on_ball: bool = False,
) -> RobotCommand:
    """
    Turns the robot on the spot to face the target orientation.

    pivot_on_ball: If True, the robot will pivot on the ball, otherwise it will pivot on its own centre.
    """
    turn = calculate_robot_velocities(
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        this_robot_data=this_robot_data,
        robot_id=robot_id,
        target_coords=(None, None),
        target_oren=target_oren,
        dribbling=dribbling,
    )

    if pivot_on_ball:
        angular_vel = turn.angular_vel
        local_left_vel = -angular_vel * (ROBOT_RADIUS * 1.8)
        turn = turn._replace(local_left_vel=local_left_vel)

    return turn


def predict_goal_y_location(
    shooter_position: Tuple[float, float], orientation: float, shoots_left: bool
) -> float:
    dx, dy = cos(orientation), sin(orientation)
    gx, _ = get_goal_centre(shoots_left)
    if dx == 0:
        return float("inf")
    t = (gx - shooter_position[0]) / dx
    return shooter_position[1] + t * dy


def calculate_defense_area(t: float, is_left: bool):
    """
    Defenders' path around the goal in the form of a rounded rectangle
    around the penalty box. Ensures defenders don't go inside the box and stay
    right on the edge. This edge of the penalty box is theirs - attackers are
    not allowed this close to the box.

    x = a * ((1 - r) * |cos(t)| * cos(t) + r * cos(t))
    y = a * ((1 - r) * |sin(t)| * sin(t) + r * sin(t))

    https://www.desmos.com/calculator/nmaf7rpmnw
    """
    assert pi / 2 <= t <= 3 * pi / 2, t
    a, r = 1.1, 2.1
    rp = a * ((1 - r) * (abs(cos(t)) * cos(t)) + r * cos(t)), a * (
        (1 - r) * (abs(sin(t)) * sin(t)) + r * sin(t)
    )
    return make_relative_to_goal_centre(rp, is_left)


def make_relative_to_goal_centre(
    p: Tuple[float, float], is_left_goal: bool
) -> Tuple[float, float]:
    if is_left_goal:
        goal_centre_x = -4.5  # TODO: Use pitch dimensions from field insteaed
        return goal_centre_x - p[0], p[1]
    else:
        goal_centre_x = 4.5
        return goal_centre_x + p[0], p[1]


EPS = 1e-5


def get_goal_centre(is_left: bool) -> Tuple[float, float]:
    return -4.5 if is_left else 4.5, 0


def relative_to(p: Tuple[float, float], o: Tuple[float, float]) -> Tuple[float, float]:
    return p[0] - o[0], p[1] - o[1]


def cross(v1, v2) -> float:
    return v1[0] * v2[1] - v1[1] * v2[0]


def ccw(v1, v2) -> int:
    # 1 if v1 is ccw of v2, -1 of v1 is cw of v2, 0 if colinear
    mag = cross(v1, v2)
    if abs(mag) < EPS:
        return 0
    if mag > 0:
        return 1
    else:
        return -1


def dot(v1, v2) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1]


def mag(v) -> float:
    return sqrt(v[0] * v[0] + v[1] * v[1])


def ang_between(v1, v2):
    res = dot(v1, v2) / (mag(v1) * mag(v2))
    if res > 0:
        res -= EPS
    else:
        res += EPS
    assert -1 <= res <= 1, f"{v1} {v2} {res}"

    return acos(res)


def step_curve(t: float, direction: int):
    STEP_SIZE = 0.0872665 * 2
    if direction == 0:
        return t
    return direction * STEP_SIZE + t


def clamp_to_goal_height(y: float) -> float:
    return max(min(y, 0.5), -0.5)


def clamp_to_parametric(t: float) -> float:
    # parametric is between pi /2 and 3pi / 2
    return min(3 * pi / 2, max(t, pi / 2))


def velocity_to_orientation(p: Tuple[float, float]) -> float:
    # Takes a velocity and converts to orientation in radians identical to robot orientation
    res = atan2(p[1], p[0])
    if res < 0:
        res += 2 * pi
    return res


def align_defenders(
    defender_position: float,
    attacker_position: Tuple[float, float],
    attacker_orientation: Optional[float],
    is_left: bool,
    env: Optional[SSLStandardEnv],
) -> Tuple[float, float]:
    """
    Calculates the next point on the defense area that the robots should go to
    defender_position is in terms of t on the parametric curve.
    """

    NO_MOVE_THRES = 1

    # Start by getting the current position of the defenders
    dx, dy = calculate_defense_area(defender_position, is_left)

    logger.debug(f"DEFENDER {dx} {dy}")
    goal_centre = get_goal_centre(is_left)

    if attacker_orientation is None:
        # In case there is no ball velocity or attackers, use centre of goal
        predicted_goal_position = goal_centre
    else:
        predicted_goal_position = goal_centre[0], clamp_to_goal_height(
            predict_goal_y_location(attacker_position, attacker_orientation, is_left)
        )

    if env:
        env.draw_line([predicted_goal_position, attacker_position], width=1, color="green")
        env.draw_line([predicted_goal_position, (dx, dy)], width=1, color="yellow")

    # Calculate the cross product relative to the predicted position of the goal

    poly = []
    for t in range(round(1000 * pi / 2), round(1000 * 3 * pi / 2) + 1):
        poly.append(calculate_defense_area(clamp_to_parametric(t / 1000), is_left))
    if env:
        env.draw_polygon(poly, width=3)

    goal_to_defender = relative_to((dx, dy), predicted_goal_position)
    goal_to_attacker = relative_to(attacker_position, predicted_goal_position)

    side = ccw(goal_to_defender, goal_to_attacker)
    angle = ang_between(goal_to_defender, goal_to_attacker)

    if is_left:
        side *= -1

    if degrees(angle) > NO_MOVE_THRES:
        # Move to the correct side
        next_t = step_curve(defender_position, side)
        logger.debug(
            f"RAW NEXT {calculate_defense_area(clamp_to_parametric(next_t), is_left)} {side} {angle}"
        )
        return calculate_defense_area(next_t, is_left)
    else:
        return calculate_defense_area(defender_position, is_left)


def to_defense_parametric(p: Tuple[float, float], is_left: bool) -> float:
    """
    Given a point p on the defenders' parametric curve (as defined by calculate_defense_area), returns the parameter value t
    which would give rise to this point.
    """

    # Ternary search the paramater, minimising the Euclidean distance between
    # the point corresponding to the predicted t and the actual point. We
    # could potentially use length along the curve (I think you can get this
    # from polar coordinates? But for a semicircle ish curve this works fine)
    lo = pi / 2
    hi = 3 * pi / 2
    EPS = 1e-6

    while (hi - lo) > EPS:
        mi1 = lo + (hi - lo) / 3
        mi2 = lo + 2 * (hi - lo) / 3

        pred1 = calculate_defense_area(mi1, is_left)
        pred2 = calculate_defense_area(mi2, is_left)

        dist1 = dist(p, pred1)
        dist2 = dist(p, pred2)
        if dist1 < dist2:
            hi = mi2
        else:
            lo = mi1

    t = lo
    return t

def goalkeep(is_left_goal: bool, game: Game, robot_id: int, pid_oren: PID, pid_trans: TwoDPID, is_yellow: bool, goalie_has_ball: bool):
    robot_data = game.get_robot_pos(is_yellow, robot_id)
    if goalie_has_ball:
        target_oren = (0 if is_left_goal else math.pi)
        print("TARGET OREN", target_oren)
        return go_to_point(pid_oren, pid_trans, robot_data, robot_id, ((-4 if is_left_goal else 4), 0), target_oren, True)

    if is_left_goal:
        target = game.predict_ball_pos_at_x(-4.5)
    else:
        target = game.predict_ball_pos_at_x(4.5)

    if not target or abs(target[1]) > 0.5:
        target = (-4.5 if is_left_goal else 4.5, 0)

    print("Going to", target)
    if target and not find_likely_enemy_shooter(game.get_robots_pos(not is_yellow), [game.ball]):
        cmd = go_to_point(
            pid_oren,
            pid_trans,
            robot_data,
            robot_id,
            target,
            face_ball(
                (robot_data.x, robot_data.y), 
                (game.ball.x, game.ball.y)
            ),
            dribbling=True
        )
    else: # TODO : Not sure if we actually need this case? 
        cmd = go_to_point(
            pid_oren,
            pid_trans,
            robot_data,
            0,
            [None, None],
            face_ball(
                (robot_data.x, robot_data.y), (game.ball.x, game.ball.y)
            ),
        )
    return cmd

def find_likely_enemy_shooter(enemy_robots, balls) -> List[RobotData]:
    ans = []
    for ball in balls:
        for er in enemy_robots:
            if dist((er.x, er.y), (ball.x, ball.y)) < 0.2:
                # Ball is close to this robot
                ans.append(er)
    return list(set(ans))

if __name__ == "__main__":
    logger.debug(f"{to_defense_parametric((3, 2), False)}")
