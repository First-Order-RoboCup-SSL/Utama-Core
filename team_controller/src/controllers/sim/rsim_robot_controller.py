from typing import Dict, Union, Optional
from entities.data.command import RobotCommand
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
import numpy as np
from numpy.typing import NDArray
from entities.data.command import RobotCommand, RobotInfo
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv


class RSimRobotController(AbstractRobotController):
    def __init__(
        self,
        is_team_yellow: bool,
        env: SSLBaseEnv,
        debug: bool = False,
    ):
        self._is_team_yellow = is_team_yellow
        self.debug = debug
        self.env = env
        self.out_packet = self._empty_command()

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue),
        """
        if self.debug:
            print(f"Sending Robot Commands")

        action = {
            "team_blue": tuple(self._empty_command()),
            "team_yellow": tuple(self.out_packet),
        }
        print(action)
        next_state, reward, terminated, truncated, reward_shaping = self.env.step(
            action
        )
        # flush out_packet
        self.out_packet = self._empty_command()

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

        Calls add_robot_command for each entered command
        """
        super().add_robot_commands(robot_commands, robot_id)

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """
        Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick_spd', 'kick_angle', 'dribbler_spd'.
        """
        action = np.array(
            [
                command.local_forward_vel,
                command.local_left_vel,
                command.angular_vel,
                command.kick_spd,
                command.kick_angle,
            ],
            dtype=np.float32,
        )
        self.out_packet[robot_id] = action

    def _empty_command(self) -> list[NDArray]:
        return [np.zeros((6,), dtype=float) for _ in range(6)]
