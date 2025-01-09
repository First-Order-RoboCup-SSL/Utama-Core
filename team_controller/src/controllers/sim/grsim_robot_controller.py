from typing import Tuple, Optional, Dict, List, Union
import warnings

from entities.data.command import RobotCommand, RobotInfo
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from team_controller.src.config.settings import (
    PID_PARAMS,
    LOCAL_HOST,
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
logger = logging.getLogger(__name__)


class GRSimRobotController(AbstractRobotController):
    def __init__(
        self,
        is_team_yellow: bool,
        address=LOCAL_HOST,
        port=(YELLOW_TEAM_SIM_PORT, BLUE_TEAM_SIM_PORT),
        debug=False,
    ):
        self.is_team_yellow = is_team_yellow

        self.out_packet = RobotControl()

        if is_team_yellow:
            self.net = network_manager.NetworkManager(address=(address, port[0]))
        else:
            self.net = network_manager.NetworkManager(address=(address, port[1]))

        self.robots_info: List[RobotInfo] = [None] * 6

        self.debug = debug

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).
        """
        logger.debug(f"Sending Robot Commands")

        data = self.net.send_command(self.out_packet, is_sim_robot_cmd=True)

        # manages the response packet that is received
        if data:
            robots_info = RobotControlResponse()
            robots_info.ParseFromString(data)
            for _, robot_info in enumerate(robots_info.feedback):
                if robot_info.HasField("dribbler_ball_contact") and robot_info.id < 6:
                    self.robots_info[robot_info.id] = RobotInfo(
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
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick_spd', 'kick_angle', 'dribbler_spd'.
        """
        robot = self.out_packet.robot_commands.add()
        robot.id = robot_id
        robot.kick_speed = command.kick_spd
        robot.kick_angle = command.kick_angle
        robot.dribbler_speed = command.dribbler_spd

        local_vel = robot.move_command.local_velocity
        local_vel.forward = command.local_forward_vel
        local_vel.left = command.local_left_vel
        local_vel.angular = command.angular_vel

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
