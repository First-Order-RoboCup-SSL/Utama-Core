from typing import Tuple, Union
from entities.data.command import RobotCommand
from entities.data.vision import RobotData
from global_utils.math_utils import rotate_vector
from motion_planning.src.pid import PID
from motion_planning.src.pid.pid import TwoDPID
import numpy as np


def calculate_robot_velocities(
    pid_oren: PID,
    pid_trans: TwoDPID,
    this_robot_data: RobotData,
    robot_id: int,
    target_coords: Union[Tuple[float, float], Tuple[float, float, float]],
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """
    Temporary PID controller for robot motion planning. Adapted from shooting controller.

    # TODO: This should eventually be stored within motion planning
    """

    current_x, current_y, current_oren = this_robot_data

    target_x, target_y = target_coords[:2]

    if target_oren is not None:
        angular_vel = pid_oren.calculate(target_oren, current_oren, robot_id, oren=True, normalize_range=np.pi)
    else:
        angular_vel = 0

    if target_x is not None and target_y is not None:
        forward_vel, left_vel = pid_trans.calculate(
            (target_x, target_y), (current_x, current_y), robot_id
        )
        forward_vel, left_vel = rotate_vector(forward_vel, left_vel, current_oren)
    else:
        forward_vel = 0
        left_vel = 0

    return RobotCommand(
        local_forward_vel=forward_vel,
        local_left_vel=left_vel,
        angular_vel=angular_vel,
        kick=0,
        chip=0,
        dribble=1 if dribbling else 0,
    )
