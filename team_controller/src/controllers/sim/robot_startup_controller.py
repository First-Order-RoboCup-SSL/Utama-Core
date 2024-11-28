import time
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Union, List

from entities.game.game import Game
from global_utils.math_utils import rotate_vector
from entities.data.command import RobotCommand
from entities.data.vision import RobotData, BallData
from team_controller.src.data import VisionDataReceiver
from team_controller.src.controllers import GRSimRobotController
from motion_planning.src.pid.pid import PID
from team_controller.src.config.settings import (
    PID_PARAMS,
)
from team_controller.src.config.starting_formation import YELLOW_START_ONE
from team_controller.src.generated_code.ssl_simulation_robot_control_pb2 import (
    RobotControl,
)

# TODO: This needs to be moved out of team_controller soon


class StartUpController:
    def __init__(
        self,
        game: Game,
        debug=False,
    ):
        self.game = game
        self.sim_robot_controller = GRSimRobotController(
            is_team_yellow=True, debug=debug
        )

        # TODO: Tune PID parameters further when going from sim to real(it works for Grsim)
        # potentially have a set tunig parameters for each robot
        self.pid_oren = PID(0.0167, 8, -8, 5, 0, 0.03, num_robots=6)
        self.pid_trans = PID(0.0167, 1.5, -1.5, 5, 0, 0.02, num_robots=6)

        self.debug = debug

    def make_decision(self):
        robots = self.game.get_robots_pos(is_yellow=True)
        balls = self.game.get_ball_pos()

        if robots and balls:
            out_packet = RobotControl()
            for robot_id, robot_data in enumerate(robots):
                if robot_data is None:
                    continue
                target_coords = YELLOW_START_ONE[robot_id]
                command = self._calculate_robot_velocities(
                    robot_id, target_coords, robots, balls, face_ball=True
                )
                self.sim_robot_controller.add_robot_commands(command, robot_id)

            if self.debug:
                print(out_packet)
            self.sim_robot_controller.send_robot_commands()
            self.sim_robot_controller.robot_has_ball(robot_id=3)

    def _calculate_robot_velocities(
        self,
        robot_id: int,
        target_coords: Union[Tuple[float, float], Tuple[float, float, float]],
        robots: List[RobotData],
        balls: List[BallData],
        face_ball=False,
    ) -> RobotCommand:
        """
        Calculates the linear and angular velocities required for a robot to move towards a specified target position
        and orientation.

        Args:
            robot_id (int): Unique identifier for the robot.
            target_coords (Tuple[float, float] | Tuple[float, float, float]): Target coordinates the robot should move towards.
                Can be a (x, y) or (x, y, orientation) tuple. If `face_ball` is True, the robot will face the ball instead of
                using the orientation value in target_coords.
            robots (Dict[int, Optional[Tuple[float, float, float]]]): All the Current coordinates of the robots sepateated
                by thier robot_id which containts a tuple (x, y, orientation).
            balls (Dict[int, Tuple[float, float, float]]): All the Coordinates of the detected balls (int) , typically (x, y, z/height in 3D space).            face_ball (bool, optional): If True, the robot will orient itself to face the ball's position. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary containing the following velocity components:
                - "id" (int): Robot identifier.
                - "xvel" (float): X-axis velocity to move towards the target.
                - "yvel" (float): Y-axis velocity to move towards the target.
                - "wvel" (float): Angular velocity to adjust the robot's orientation.

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
                target_oren, current_oren, robot_id, oren=True
            )
        else:
            angular_vel = 0

        if target_x != None and target_y != None:
            left_vel = self.pid_trans.calculate(
                target_y, current_y, robot_id, normalize_range=3
            )
            forward_vel = self.pid_trans.calculate(
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
            dribbler_spd=0,
        )
