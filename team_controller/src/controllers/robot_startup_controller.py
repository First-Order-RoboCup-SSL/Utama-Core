import time
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Union

from global_utils.math_utils import rotate_vector
from entities.data.command import RobotSimCommand
from team_controller.src.data.vision_receiver import VisionDataReceiver
from team_controller.src.controllers.sim_robot_controller import SimRobotController
from motion_planning.src.pid.pid import PID
from team_controller.src.config.settings import (
    PID_PARAMS,
    YELLOW_START,
)

# TODO: To be moved to a High-level Descision making repo

class StartUpController:
    def __init__(
        self,
        vision_receiver: VisionDataReceiver,
        debug=False,
    ):
        self.vision_receiver = vision_receiver
        self.sim_robot_controller = SimRobotController(is_team_yellow=True, debug=debug)

        # TODO: Tune PID parameters further when going from sim to real(it works for Grsim)
        # potentially have a set tunig parameters for each robot
        self.pid_oren = PID(0.0167, 8, -8, 5, 0, 0.03, num_robots=6)
        self.pid_trans = PID(0.0167, 1.5, -1.5, 5, 0, 0.02, num_robots=6)

        self.lock = threading.Lock()

        self.debug = debug

    def startup(self):
        while True:
            start_time = time.time()

            robots, balls = self._get_positions()

            if robots and balls:
                for robot_id, robot_data in enumerate(robots):
                    if robot_data is None:
                        continue
                    target_coords = YELLOW_START[robot_id]
                    command = self._calculate_robot_velocities(
                        robot_id, target_coords, robots, balls, face_ball=True
                    )
                    self.sim_robot_controller.add_robot_commands(command, robot_id)
                    
                self.sim_robot_controller.send_robot_commands()
                self.sim_robot_controller.robot_has_ball(robot_id=3)

            time_to_sleep = max(0, 0.0167 - (time.time() - start_time))
            time.sleep(time_to_sleep)

    def _get_positions(self) -> tuple:
        # Fetch the latest positions of robots and balls with thread locking.
        with self.lock:
            robots = self.vision_receiver.get_robots_pos(is_yellow=True)
            balls = self.vision_receiver.get_ball_pos()
        return robots, balls

    def _calculate_robot_velocities(
            self,
            robot_id: int,
            target_coords: Union[Tuple[float, float], Tuple[float, float, float]],
            robots: Dict[int, Optional[Tuple[float, float, float]]],
            balls: Dict[int, Tuple[float, float, float]],
            face_ball=False,
        ) -> RobotSimCommand:
            """
                    Calculates the linear and angular velocities required for a robot to move towards a specified target position
                    and orientation.

                    Args:
                        robot_id (int): The identifier of the robot.
                        target_coords (Tuple[float, float] or Tuple[float, float, float]): The target position and orientation.
                        robots (Dict[int, Optional[Tuple[float, float, float]]]): A dictionary containing the positions of all robots.
                        balls (Dict[int, Tuple[float, float, float]]): A dictionary containing the positions of all balls.
                        face_ball (bool, optional): Whether the robot should face the ball. Defaults to False.

                    Returns:
                        RobotSimCommand: A named tuple containing the linear and angular velocities, kick speed, kick angle, and dribbler speed.
                        
                    The method uses PID controllers to calculate velocities for linear and angular movement. If `face_ball` is set,
                    the robot will calculate the angular velocity to face the ball. The resulting x and y velocities are rotated to align
                    with the robot's current orientation.
            """
            
            # Get current positions
            if balls[0] and robots[robot_id]:
                ball_x, ball_y, ball_z = balls[0]
                current_x, current_y, current_oren = robots[robot_id]

            target_x, target_y = target_coords[:2]

            if face_ball:
                target_oren = np.atan2(ball_y - current_y, ball_x - current_x)
            elif not face_ball and len(target_coords) == 3:
                target_oren = target_coords[2]

            # print(f"\nRobot {robot_id} current position: ({current_x:.3f}, {current_y:.3f}, {current_oren:.3f})")
            # print(f"Robot {robot_id} target position: ({target_x:.3f}, {target_y:.3f}, {target_oren:.3f})")

            if target_oren != None:
                angular_vel = self.pid_oren.calculate(
                    target_oren, current_oren, robot_id, oren=True, normalize_range=np.pi/2
                )

            if target_x != None and target_y != None:
                left_vel = self.pid_trans.calculate(
                    target_y, current_y, robot_id, normalize_range=3000
                )
                forward_vel = self.pid_trans.calculate(
                    target_x, current_x, robot_id, normalize_range=4500
                )

                forward_vel, left_vel = rotate_vector(
                    forward_vel, left_vel, current_oren
                )

            # print(f"Output: {forward_vel}, {left_vel}, {angular_vel}")
            return RobotSimCommand(local_forward_vel=forward_vel, local_left_vel=left_vel, angular_vel=angular_vel, kick_spd=0, kick_angle=0, dribbler_spd=0)
