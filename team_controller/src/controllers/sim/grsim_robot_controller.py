from typing import Tuple, Optional, Dict, List, Union
import warnings
import logging

from entities.data.command import RobotCommand, RobotVelCommand, RobotResponse
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from config.settings import (
    KICK_SPD,
    DRIBBLE_SPD,
    CHIP_ANGLE,
    PID_PARAMS,
    LOCAL_HOST,
    TIMESTEP,
    YELLOW_TEAM_SIM_PORT,
    BLUE_TEAM_SIM_PORT,
)
from team_controller.src.utils import network_manager

from team_controller.src.generated_code.ssl_simulation_robot_control_pb2 import (
    RobotControl,
)

from team_controller.src.generated_code.ssl_simulation_robot_feedback_pb2 import (
    RobotControlResponse,
    RobotFeedback,
)

import logging
import time


logger = logging.getLogger(__name__)


class GRSimRobotController(AbstractRobotController):
    def __init__(
        self,
        is_team_yellow: bool,
        address=LOCAL_HOST,
        port=(YELLOW_TEAM_SIM_PORT, BLUE_TEAM_SIM_PORT),
    ):
        self.is_team_yellow = is_team_yellow

        self.out_packet = RobotControl()

        if is_team_yellow:
            self.net = network_manager.NetworkManager(address=(address, port[0]))
        else:
            self.net = network_manager.NetworkManager(address=(address, port[1]))

        self.robots_info: List[RobotResponse] = [None] * 6
        self.net_diff_sum = 0
        self.net_diff_total = 0

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).
        """
        logger.debug(f"Sending Robot Commands")

        net_start = time.time()
        data = self.net.send_command(self.out_packet, is_sim_robot_cmd=True)
        net_end = time.time()
        net_diff = net_end - net_start

        self.net_diff_sum += net_diff
        self.net_diff_total += 1

        average_net_diff = self.net_diff_sum / self.net_diff_total
        logger.log(logging.WARNING if net_diff > TIMESTEP / 2 else logging.INFO,
                   "Sending commands to GRSIM took %f",
                   net_diff)
        logger.log(
            logging.WARNING if net_diff > TIMESTEP / 2 else logging.INFO,
            "GRSIM Command Send Time Avg: %f",
            average_net_diff
        )

        st = time.time()
        # manages the response packet that is received
        if data:
            robots_info = RobotControlResponse()
            robots_info.ParseFromString(data)
            for _, robot_info in enumerate(robots_info.feedback):
                if robot_info.HasField("dribbler_ball_contact") and robot_info.id < 6:
                    self.robots_info[robot_info.id] = RobotResponse(
                        robot_info.dribbler_ball_contact
                    )
                elif (
                    robot_info.HasField("dribbler_ball_contact") and robot_info.id >= 6
                ):
                    warnings.warn(
                        "Invalid robot info received, robot id >= 6", SyntaxWarning
                    )
        self.out_packet.Clear()
    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """
        Adds robot commands to the out_packet.

        Args:
            robot_commands (Union[RobotCommand, Dict[int, RobotCommand]]): A single RobotCommand or a dictionary of RobotCommand with robot_id as the key.
            robot_id (Optional[int]): The ID of the robot which is ONLY used when adding one Robot command. Defaults to None.

        Raises:
            SyntaxWarning: If invalid hyperparameters are passed to the function.

        Calls _add_robot_command for each entered command
        """
        super().add_robot_commands(robot_commands, robot_id)

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """
        Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick', 'chip', 'dribble'.
        """
        robot = self.out_packet.robot_commands.add()
        robot.id = robot_id
        robot.kick_speed = KICK_SPD if (command.kick or command.chip) > 0 else 0
        robot.kick_angle = CHIP_ANGLE if command.chip > 0 else 0
        robot.dribbler_speed = DRIBBLE_SPD if command.dribble > 0 else 0
        # print(robot)

        local_vel = robot.move_command.local_velocity
        local_vel.forward = command.local_forward_vel
        local_vel.left = command.local_left_vel
        local_vel.angular = command.angular_vel

    def _add_robot_wheel_vel_command(
        self, command: RobotVelCommand, robot_id: int
    ) -> None:
        """
        Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick', 'chip', 'dribble'.
        """
        robot = self.out_packet.robot_commands.add()
        robot.id = robot_id
        robot.kick_speed = KICK_SPD if (command.kick or command.chip) > 0 else 0
        robot.kick_angle = CHIP_ANGLE if command.chip > 0 else 0
        robot.dribbler_speed = DRIBBLE_SPD if command.dribble > 0 else 0

        wheel_vel = robot.move_command.wheel_velocity
        wheel_vel.front_right = command.front_right
        wheel_vel.front_left = command.front_left
        wheel_vel.back_right = command.back_right
        wheel_vel.back_left = command.back_left

    def robot_has_ball(self, robot_id: int) -> bool:
        """
        Checks if the specified robot has the ball.

        Args:
            robot_id (int): The ID of the robot.

        Returns:
            bool: True if the robot has the ball, False otherwise.
        """
        if self.robots_info[robot_id] is None:
            return False

        if self.robots_info[robot_id].has_ball:
            logger.debug(f"Robot: {robot_id}: HAS the Ball")
            return True
        else:
            return False
