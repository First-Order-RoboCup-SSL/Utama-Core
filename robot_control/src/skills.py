import numpy as np
from typing import Optional, Tuple

from entities.data.command import RobotCommand
from entities.data.vision import BallData, RobotData

from motion_planning.src.pid import PID


def kick_ball() -> RobotCommand:
    return RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick_spd=3,
        kick_angle=0,
        dribbler_spd=0,
    )


from motion_planning.src.pid.pid import TwoDPID
from robot_control.src.utils.motion_planning_utils import calculate_robot_velocities


# TODO: this pid system is v clumsy, and will eventually be streamlined with robot object
# Ideally, we should have one PID for each robot and not have one giant PID for all robots
def go_to_ball(
    pid_oren: PID,
    pid_trans: TwoDPID,
    this_robot_data: RobotData,
    robot_id: int,
    ball_data: BallData,
) -> RobotCommand:
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

def face_ball(current: Tuple[float, float], ball:Tuple[float, float]) -> float:
    return  np.arctan2(
        ball[1] - current[1], ball[0] - current[0]
    )

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
) -> RobotCommand:
    return calculate_robot_velocities(
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        this_robot_data=this_robot_data,
        robot_id=robot_id,
        target_coords=(None, None),
        target_oren=target_oren,
        dribbling=dribbling,
    )

from math import atan, atan2, dist, sqrt, cos, sin, pi, acos, degrees, asin
RADIUS = sqrt(2)+0.1


def predict_goal_y_location(shooter_position: Tuple[float, float], orientation:float, shoots_left: bool) -> float:
    dx, dy = cos(orientation), sin(orientation)
    gx,_ = get_goal_centre(shoots_left)
    if dx == 0:
        return float("inf")
    t = (gx-shooter_position[0]) / dx
    return shooter_position[1] + t*dy

import numpy as np
def calculate_defense_area(t: float, is_left: bool):
    """ Semi circle around goal with radius sqrt(2) metres, t is parametric 
    x = r cos t, y = r sin t, pi/2 <= t <= 3pi/2 
    
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2 / n)
    y = b * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2 / n)
    """
    assert pi/2 <= t <= 3*pi/2, t
    a,r = 1.1, 2.2
    rp = a * ((1-r)*(abs(cos(t))*cos(t))+r*cos(t)), a * ((1-r)*(abs(sin(t))*sin(t))+r*sin(t))
    return make_relative_to_goal_centre(rp, is_left)


def make_relative_to_goal_centre(p: Tuple[float, float], is_left_goal: bool) -> Tuple[float, float]:
    if is_left_goal:
        goal_centre_x = -4.5
        return goal_centre_x-p[0], p[1]
    else:
        goal_centre_x = 4.5
        return goal_centre_x+p[0], p[1]


EPS = 1e-5
def get_goal_centre(is_left: bool) -> Tuple[float, float]:
    return -4.5 if is_left else 4.5, 0

def relative_to(p: Tuple[float, float], o: Tuple[float, float]) -> Tuple[float, float]:
    return p[0]-o[0], p[1]-o[1]

def cross(v1, v2) -> float:
    return v1[0]*v2[1]-v1[1]*v2[0]

def ccw(v1, v2) -> int :
    # 1 if v1 is ccw of v2, -1 of v1 is cw of v2, 0 if colinear
    mag = cross(v1, v2)
    if abs(mag) < EPS:
        return 0
    if mag > 0:
        return 1
    else:
        return -1

def dot(v1, v2) -> float:
    return v1[0]*v2[0]+v1[1]*v2[1]

def mag(v) -> float:
    return sqrt(v[0]*v[0]+v[1]*v[1])

def ang_between(v1, v2):
    print("DOT", dot(v1, v2))
    res = dot(v1,v2) / (mag(v1)*mag(v2))
    if res > 0:
        res -= EPS
    else:
        res += EPS
    assert -1 <=  res <= 1, f"{v1} {v2} {res}"

    return acos(res)



def step_curve(t: float, direction:int):
    STEP_SIZE = 0.0872665
    if direction == 0:
        return t
    return direction*STEP_SIZE + t

def clamp_to_goal_height(y: float) -> float:
    return max(min(y, 0.5), -0.5)

def clamp_to_parametric(t:float) -> float:
    # parametric is between pi /2 and 3pi / 2
    return min(3*pi / 2, max(t,pi/2))

def velocity_to_orientation(p: Tuple[float, float]) -> float:
    # Takes a velocity and converts to orientation in radians identical to robot orientation
    res = atan2(p[1], p[0])
    if res < 0:
        res += 2*pi
    return res

def align_defenders(defender_position: float, attacker_position: Tuple[float, float], attacker_orientation: Optional[float], is_left: bool, env) -> Tuple[float, float]:
    # defender_position is in terms of t, the parametric 
    NO_MOVE_THRES = 1
    # Calculates the next point on the defense area that the robots should go to
    dx, dy = calculate_defense_area(defender_position, is_left)
    print("DEFENDER", dx, dy)
    goal_centre = get_goal_centre(is_left)

    if attacker_orientation is None:
        # In case there is no ball velocity or attackers, use centre of goal 
        predicted_goal_position = goal_centre
    else:
        predicted_goal_position = goal_centre[0], clamp_to_goal_height(predict_goal_y_location(attacker_position, attacker_orientation, is_left))

    
    env.draw_line([predicted_goal_position, attacker_position], width=1, color="green")
    # Calculate the cross product relative to the predicted position of the goal 

    poly = []
    for t in range(round(1000 * pi/2), round(1000*3*pi / 2)+1):
        poly.append(calculate_defense_area(clamp_to_parametric(t/1000), is_left))
    env.draw_polygon(poly, width=3)

    goal_to_defender = relative_to((dx, dy), predicted_goal_position)
    goal_to_attacker = relative_to(attacker_position, predicted_goal_position)
    print(goal_to_defender, goal_to_attacker)

    side = ccw(goal_to_defender, goal_to_attacker)
    angle = ang_between(goal_to_defender, goal_to_attacker)
    
    print(side, degrees(angle), "HELLO ")

    if is_left:
        side *= -1

    if degrees(angle) > NO_MOVE_THRES:
        # Move to the correct side 
        next_t = clamp_to_parametric(step_curve(defender_position, side))
        print("RAW NEXT", calculate_defense_area(next_t, is_left), side, angle, next_t, defender_position)
        return calculate_defense_area(next_t, is_left)
    else:
        return calculate_defense_area(defender_position, is_left)



def to_defense_parametric(p: Tuple[float, float], is_left: bool) -> float:
    gp = get_goal_centre(is_left)
    rel_goal = relative_to(p, gp)
    ang = ang_between(rel_goal, (0,1))
    print(degrees(ang))
    if ang > pi:
        ang = 2*pi-ang
    ang += pi / 2
    print(calculate_defense_area(ang, is_left))
    return ang


if __name__=="__main__":
    # print("NEXT POS: ", align_defenders(pi, (2.5, 2), False))
    print(to_defense_parametric((3, 2), False))

    
