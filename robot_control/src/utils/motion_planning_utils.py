from typing import Tuple, Union
from entities.data.command import RobotCommand
from entities.data.vision import RobotData
from global_utils.math_utils import rotate_vector
from motion_planning.src.pid import PID


def calculate_robot_velocities(
    pid_oren: PID,
    pid_trans: PID,
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

    # print(f"\nRobot {robot_id} current position: ({current_x:.3f}, {current_y:.3f}, {current_oren:.3f})")
    # print(f"Robot {robot_id} target position: ({target_x:.3f}, {target_y:.3f}, {target_oren:.3f})")

    if target_oren is not None:
        angular_vel = pid_oren.calculate(target_oren, current_oren, robot_id, oren=True)
    else:
        angular_vel = 0

    if target_x is not None and target_y is not None:
        left_vel = pid_trans.calculate(target_y, current_y, robot_id, normalize_range=3)
        forward_vel = pid_trans.calculate(
            target_x, current_x, robot_id, normalize_range=4.5
        )

        forward_vel, left_vel = rotate_vector(forward_vel, left_vel, current_oren)
    else:
        forward_vel = 0
        left_vel = 0
    # print(f"Output: {forward_vel}, {left_vel}, {angular_vel}")
    return RobotCommand(
        local_forward_vel=forward_vel,
        local_left_vel=left_vel,
        angular_vel=angular_vel,
        kick_spd=0,
        kick_angle=0,
        dribbler_spd=3 if dribbling else 0,
    )
