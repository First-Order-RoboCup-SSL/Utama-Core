import numpy as np
from typing import Tuple
from time import sleep

from entities.data.command import RobotCommand, RobotInfo
from entities.data.vision import BallData, RobotData

from motion_planning.src.pid import PID
from global_utils.math_utils import distance

from robot_control.src.utils.motion_planning_utils import calculate_robot_velocities
from robot_control.src import skills

def dribble_to_target(
    pid_oren: PID,
    pid_trans: PID,
    this_robot_data: RobotData,
    robot_id: int,
    target_coords: Tuple[float, float],
    dribble_speed: float = 3.0,
    tolerance: float = 0.1,
    ):
    """
    Function to dribble the ball to a target location using repeated cycles of releasing, 
    repositioning, and reabsorbing the ball.

    Parameters:
        pid_oren (PID): PID controller for orientation.
        pid_trans (PID): PID controller for translation.
        this_robot_data (RobotData): Current robot state (x, y, orientation).
        robot_id (int): ID of the robot.
        target_coords (Tuple[float, float]): Target x, y coordinates.
        dribble_speed (float): Speed of the dribbler. Defaults to 3.0.
        tolerance (float): Tolerance for reaching the target. Defaults to 0.1 meters.
    Returns:
        None
    """
    current_x, current_y, current_oren = this_robot_data
    target_x, target_y = target_coords

    while distance((current_x, current_y), (target_x, target_y)) > tolerance:
        # Step 1: Set dribbler speed negative to take the ball
        if (RobotInfo.stop_current_command): return
        yield RobotCommand(
            local_forward_vel=0,
            local_left_vel=0,
            angular_vel=0,
            kick_spd=0,
            kick_angle=0,
            dribbler_spd=-dribble_speed,
        )

        if (RobotInfo.stop_current_command): return
        sleep(0.1) # Wait for the ball to be absorbed

        # Step 2: Stop dribbler
        if (RobotInfo.stop_current_command): return
        yield RobotCommand(
            local_forward_vel=0,
            local_left_vel=0,
            angular_vel=0,
            kick_spd=0,
            kick_angle=0,
            dribbler_spd=0,
        )

        # Step 3: Turn to target point
        if (RobotInfo.stop_current_command): return
        target_oren = np.arctan2(target_y - current_y, target_x - current_x)
        yield calculate_robot_velocities(
            pid_oren=pid_oren,
            pid_trans=pid_trans,
            this_robot_data=this_robot_data,
            robot_id=robot_id,
            target_coords=(None, None),
            target_oren=target_oren,
        )

        # Step 4: Move forward 0.8 meters, if the rule allows
        if (RobotInfo.stop_current_command): return
        forward_distance = 0.8
        move_target_x = current_x + forward_distance * np.cos(current_oren)
        move_target_y = current_y + forward_distance * np.sin(current_oren)
        yield calculate_robot_velocities(
            pid_oren=pid_oren,
            pid_trans=pid_trans,
            this_robot_data=this_robot_data,
            robot_id=robot_id,
            target_coords=(move_target_x, move_target_y),
            target_oren=current_oren,
        )

        # Step 5: Release the ball slightly
        if (RobotInfo.stop_current_command): return
        yield RobotCommand(
            local_forward_vel=0,
            local_left_vel=0,
            angular_vel=0,
            kick_spd=0.5,
            kick_angle=0,
            dribbler_spd=0,
        )

        # Step 6: Move towards the ball to reabsorb it
        if (RobotInfo.stop_current_command): return
        ball_coords = (move_target_x, move_target_y)  # Approximate ball position
        yield calculate_robot_velocities(
            pid_oren=pid_oren,
            pid_trans=pid_trans,
            this_robot_data=this_robot_data,
            robot_id=robot_id,
            target_coords=ball_coords,
            target_oren=current_oren,
            dribbling=True,
        )

        # Step 7: Update current position (simulate robot movement)
        current_x, current_y = move_target_x, move_target_y

    # Final stop command when target is reached
    yield RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick_spd=0,
        kick_angle=0,
        dribbler_spd=0,
    )
