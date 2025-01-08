import numpy as np
from typing import Tuple

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
    print("target oren", target_oren)
    return calculate_robot_velocities(
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        this_robot_data=this_robot_data,
        robot_id=robot_id,
        target_coords=ball_data,
        target_oren=target_oren,
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
