### DEPRECATED ###

import time
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Union

from global_utils.math_utils import rotate_vector
from team_controller.src.data.vision_receiver import VisionDataReceiver
from motion_planning.src.pid.pid import PID
from team_controller.src.config.settings import (
    LOCAL_HOST,
    YELLOW_TEAM_SIM_PORT,
    BLUE_TEAM_SIM_PORT,
)
from team_controller.src.utils import network_manager
from robot_control.src.find_best_shot import ray_casting, find_best_shot

from team_controller.src.generated_code.ssl_simulation_robot_control_pb2 import (
    RobotControl,
)

import logging
logger = logging.getLogger(__name__)


class ShootingController:
    def __init__(
        self,
        shooter_id,
        goal_x,
        goal_y1,
        goal_y2,
        vision_receiver: VisionDataReceiver,
        address=LOCAL_HOST,
        port=(YELLOW_TEAM_SIM_PORT, BLUE_TEAM_SIM_PORT),
    ):
        self.vision_receiver = vision_receiver
        self.goal_x = goal_x
        self.goal_y1 = goal_y1
        self.goal_y2 = goal_y2

        self.net = network_manager.NetworkManager(address=(address, port[0]))

        # TODO: Tune PID parameters further when going from sim to real(it works for Grsim)
        # potentially have a set tunig parameters for each robot
        self.pid_oren = PID(0.0167, 8, -8, 4.5, 0, 0, num_robots=6)
        self.pid_trans = PID(0.0167, 1.5, -1.5, 4.5, 0, 0, num_robots=6)

        self.lock = threading.Lock()

        self.shooter_id = shooter_id

    def kick_ball(self):
        logger.debug("kicked the ball")
        out_packet = self._add_kick_command(self.shooter_id, 4)
        logger.debug(str(out_packet))
        self.net.send_command(out_packet)

    def approach_ball(self):

        while True:
            start_time = time.time()

            robots, enemy_robots, balls = self._get_positions()

            if robots and balls:
                out_packet = RobotControl()
                shadows = ray_casting(
                    balls[0], enemy_robots, self.goal_x, self.goal_y1, self.goal_y2
                )
                best_shot = find_best_shot(shadows, self.goal_y1, self.goal_y2)

                shot_orientation = np.pi + np.atan(
                    (best_shot - balls[0].y) / (self.goal_x - balls[0].x)
                )

                robot_data = (
                    robots[self.shooter_id] if self.shooter_id < len(robots) else None
                )
                if robot_data is not None:
                    target_coords = (
                        balls[0].x - 15,
                        balls[0].y - 15,
                        shot_orientation,
                    )
                    command = self._calculate_robot_velocities(
                        self.shooter_id, target_coords, robots, balls, face_ball=False
                    )
                    self._add_robot_command(out_packet, command)

                    # very arbitrary condition to start shooting. needs to be changed
                    if (
                        (robot_data.x - balls[0].x) ** 2
                        + (robot_data.y - balls[0].y) ** 2
                    ) < 30000:
                        self.kick_ball()

                logger.debug(str(out_packet))
                self.net.send_command(out_packet)

            time_to_sleep = max(0, 0.0167 - (time.time() - start_time))
            time.sleep(time_to_sleep)

    def _add_kick_command(robot_id, kick_speed, kick_angle=0.0, dribbler_speed=1.0):
        robot_control = RobotControl()
        command = robot_control.robot_commands.add()
        command.id = 3
        command.kick_speed = kick_speed
        command.kick_angle = kick_angle
        command.dribbler_speed = dribbler_speed
        return robot_control

    # Functions copied from team_controller

    def _get_positions(self) -> tuple:
        # Fetch the latest positions of robots and balls with thread locking.
        with self.lock:
            robots = self.vision_receiver.get_robots_pos(is_yellow=True)
            enemy_robots = self.vision_receiver.get_robots_pos(is_yellow=False)
            balls = self.vision_receiver.get_ball_pos()
        return robots, enemy_robots, balls

    def _calculate_robot_velocities(
        self,
        robot_id: int,
        target_coords: Union[Tuple[float, float], Tuple[float, float, float]],
        robots: Dict[int, Optional[Tuple[float, float, float]]],
        balls: Dict[int, Tuple[float, float, float]],
        face_ball=False,
    ) -> Dict[str, float]:
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

        out = {"id": robot_id, "xvel": 0, "yvel": 0, "wvel": 0}

        # Get current positions
        if balls[0] and robots[robot_id]:
            ball_x, ball_y, ball_z = balls[0]
            current_x, current_y, current_oren = robots[robot_id]

        target_x, target_y = target_coords[:2]

        if face_ball:
            target_oren = np.atan2(ball_y - current_y, ball_x - current_x)
        elif not face_ball and len(target_coords) == 3:
            target_oren = target_coords[2]

        logger.debug(f"\nRobot {robot_id} current position: ({current_x:.3f}, {current_y:.3f}, {current_oren:.3f})")
        logger.debug(f"Robot {robot_id} target position: ({target_x:.3f}, {target_y:.3f}, {target_oren:.3f})")

        if target_oren != None:
            out["wvel"] = self.pid_oren.calculate(
                target_oren, current_oren, robot_id, oren=True
            )

        if target_x != None and target_y != None:
            out["yvel"] = self.pid_trans.calculate(
                target_y, current_y, robot_id, normalize_range=3000
            )
            out["xvel"] = self.pid_trans.calculate(
                target_x, current_x, robot_id, normalize_range=4500
            )

            out["xvel"], out["yvel"] = rotate_vector(
                out["xvel"], out["yvel"], current_oren
            )

        return out

    def _add_robot_command(self, out_packet, command) -> None:
        robot = out_packet.robot_commands.add()
        robot.id = command["id"]
        local_vel = robot.move_command.local_velocity
        local_vel.forward = command["xvel"]
        local_vel.left = command["yvel"]
        local_vel.angular = command["wvel"]
