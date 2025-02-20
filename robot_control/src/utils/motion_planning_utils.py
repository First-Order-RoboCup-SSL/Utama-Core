from typing import Tuple, Union
from entities.data.command import RobotCommand
from entities.data.vision import RobotData
from global_utils.math_utils import rotate_vector
from motion_planning.src.pid import PID
from motion_planning.src.pid.pid import TwoDPID
import numpy as np

### Note: Fully Commit this file Thanks :) ###

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

    if target_x is not None and target_y is not None:
        global_x, global_y = pid_trans.calculate(
            (target_x, target_y), (current_x, current_y), robot_id
        )
    else:
        global_x = 0
        global_y = 0
    
    forward_vel, left_vel = rotate_vector(global_x, global_y, current_oren)

    # if forward_vel and left_vel:
    
    #     print(f"Resultant vel: {resultant_vel}")
        
    #     travel_dist = resultant_vel*(1/60)
        
    #     dx = travel_dist * global_x/resultant_vel
    #     dy = travel_dist * global_y/resultant_vel
    
    if target_oren is not None:
        angular_vel = pid_oren.calculate(target_oren, current_oren, robot_id, oren=True, normalize_range=np.pi)
    else:
        angular_vel = 0
        

    #     resultant_vel = np.linalg.norm([global_x, global_y])       
    #     oren_offset = (angular_vel / 2) * resultant_vel / pid_trans.max_velocity # scales linearly [0, 1]
        
    #     if left_vel > 0:
    #         oren_offset *= -1
    #     elif left_vel < 0:
    #         oren_offset *= 1
    #     else:
    #         oren_offset = 0
    #     # print(f"Oren offset: {oren_offset}, angular vel: {angular_vel}, resultant vel: {resultant_vel}")  
    #     angular_vel += oren_offset
    #     # print(f"diff: {oren_offset*60}, New angular vel: {angular_vel}, direction: {left_vel}")

    return RobotCommand(
        local_forward_vel=forward_vel,
        local_left_vel=left_vel,
        angular_vel=angular_vel,
        kick=0,
        chip=0,
        dribble=1 if dribbling else 0,
    )
