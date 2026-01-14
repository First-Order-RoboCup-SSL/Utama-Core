import logging
import warnings
from typing import Dict, List, Optional, Union

from utama_core.config.physical_constants import MAX_ROBOTS
from utama_core.config.robot_params import GRSIM_PARAMS
from utama_core.config.settings import (
    BLUE_TEAM_SIM_PORT,
    LOCAL_HOST,
    YELLOW_TEAM_SIM_PORT,
)
from utama_core.entities.data.command import (
    RobotCommand,
    RobotResponse,
    RobotVelCommand,
)
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from utama_core.team_controller.src.generated_code.ssl_simulation_robot_control_pb2 import (
    RobotControl,
)
from utama_core.team_controller.src.generated_code.ssl_simulation_robot_feedback_pb2 import (
    RobotControlResponse,
)
from utama_core.team_controller.src.utils import network_manager

logger = logging.getLogger(__name__)

KICK_SPD = GRSIM_PARAMS.KICK_SPD
CHIP_ANGLE = GRSIM_PARAMS.CHIP_ANGLE
DRIBBLE_SPD = GRSIM_PARAMS.DRIBBLE_SPD


class GRSimRobotController(AbstractRobotController):
    def __init__(
        self,
        is_team_yellow: bool,
        n_friendly: int,
        address=LOCAL_HOST,
        port=(YELLOW_TEAM_SIM_PORT, BLUE_TEAM_SIM_PORT),
    ):
        super().__init__(is_team_yellow, n_friendly)

        self.out_packet = RobotControl()

        if is_team_yellow:
            self.net = network_manager.NetworkManager(address=(address, port[0]))
        else:
            self.net = network_manager.NetworkManager(address=(address, port[1]))

        self.net_diff_sum = 0
        self.net_diff_total = 0

    def send_robot_commands(self) -> None:
        """Sends the robot commands to the appropriate team (yellow or blue)."""
        # logger.debug("Sending Robot Commands ...")

        # net_start = time.time()
        data = self.net.send_command(self.out_packet, is_sim_robot_cmd=True)
        self.out_packet.Clear()
        # net_end = time.time()
        # net_diff = net_end - net_start

        # self.net_diff_sum += net_diff
        # self.net_diff_total += 1

        # average_net_diff = self.net_diff_sum / self.net_diff_total
        # logger.log(logging.WARNING if net_diff > TIMESTEP / 2 else logging.INFO,
        #            "Sending commands to GRSIM took %f",
        #            net_diff)
        # logger.log(
        #     logging.WARNING if net_diff > TIMESTEP / 2 else logging.INFO,
        #     "GRSIM Command Send Time Avg: %f",
        #     average_net_diff
        # )

        # manages the response packet that is received
        if data:
            self._update_robot_info(data)

    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """Adds robot commands to the out_packet.

        Args:
            robot_commands (Union[RobotCommand, Dict[int, RobotCommand]]): A single RobotCommand or a dictionary of RobotCommand with robot_id as the key.
            robot_id (Optional[int]): The ID of the robot which is ONLY used when adding one Robot command. Defaults to None.

        Raises:
            SyntaxWarning: If invalid hyperparameters are passed to the function.

        Calls _add_robot_command for each entered command
        """
        super().add_robot_commands(robot_commands, robot_id)

    def _update_robot_info(self, data: bytes) -> None:
        robots_info: List[RobotResponse] = []
        robots_response = RobotControlResponse()
        robots_response.ParseFromString(data)
        for _, robot_info in enumerate(robots_response.feedback):
            if robot_info.HasField("dribbler_ball_contact") and robot_info.id >= MAX_ROBOTS:
                warnings.warn("Invalid robot info received, robot id >= 6", SyntaxWarning)
            elif robot_info.id < self._n_friendly:
                robot_resp = RobotResponse(id=robot_info.id, has_ball=robot_info.dribbler_ball_contact)
                robots_info.append(robot_resp)
            # ignore robot_info for robots that are not expected (ie deactivated since the start of the game)
        self._robots_info.append(robots_info)

    def get_robots_responses(self) -> Optional[RobotResponse]:
        if self._robots_info is None or len(self._robots_info) == 0:
            for i in range(self._n_friendly):
                self.add_robot_commands(RobotCommand(0, 0, 0, False, False, False), i)
            data = self.net.send_command(self.out_packet, is_sim_robot_cmd=True)

            self.out_packet.Clear()

            if data:
                self._update_robot_info(data)
            else:
                logger.warning("No robot responses received from GRSIM.")
                return None

        return self._robots_info.popleft()

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick', 'chip', 'dribble'.
        """
        robot = self.out_packet.robot_commands.add()
        robot.id = robot_id
        robot.kick_speed = KICK_SPD if (command.kick or command.chip) > 0 else 0
        robot.kick_angle = CHIP_ANGLE if command.chip > 0 else 0
        robot.dribbler_speed = DRIBBLE_SPD if command.dribble > 0 else 0

        local_vel = robot.move_command.local_velocity
        local_vel.forward = command.local_forward_vel
        local_vel.left = command.local_left_vel
        local_vel.angular = command.angular_vel

    def _add_robot_wheel_vel_command(self, command: RobotVelCommand, robot_id: int) -> None:
        """Adds a robot command to the out_packet.

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
