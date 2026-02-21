import abc
from collections import deque
from typing import Deque, Dict, List, Optional, Union

from utama_core.entities.data.command import RobotCommand, RobotResponse


class AbstractRobotController:
    def __init__(self, is_team_yellow: bool, n_friendly: int):
        self._is_team_yellow = is_team_yellow
        self._n_friendly = n_friendly
        self._robots_info: Deque[List[RobotResponse]] = deque([], maxlen=1)

    @abc.abstractmethod
    def send_robot_commands(self) -> None:
        """Sends the robot commands to the appropriate team (yellow or blue)."""
        pass

    @abc.abstractmethod
    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """Adds robot commands to self.out_packet.

        Args:
            robot_commands (Union[RobotCommand, Dict[int, RobotCommand]]): A single RobotCommand or a dictionary of RobotCommand with robot_id as the key.
            robot_id (Optional[int]): The ID of the robot which is ONLY used when adding one Robot command. Defaults to None.

        Raises:
            SyntaxWarning: If invalid hyperparameters are passed to the function.
        """
        if isinstance(robot_commands, RobotCommand):
            if robot_id is None:
                raise ValueError("robot_id cannot be None type!")
            self._add_robot_command(robot_commands, robot_id)
        elif isinstance(robot_commands, dict):
            for robot_id, command in robot_commands.items():
                self._add_robot_command(command, robot_id)
        else:
            raise TypeError(
                "robot_commands must be a RobotCommand or a dictionary of RobotCommand with robot_id as the key."
            )

    @abc.abstractmethod
    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """Adds a robot command to self.out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick', 'chip', 'dribble'.
        """
        ...

    @abc.abstractmethod
    def get_robots_responses(self) -> Optional[List[RobotResponse]]:
        """
        Pull the latest robot response.

        Returns:
            A list of RobotResponse objects (each containing the robot ID
            and its ball-contact state), or None if no response was received.
        """
        pass
