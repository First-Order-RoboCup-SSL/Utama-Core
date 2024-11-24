import abc
import warnings
from entities.data.command import RobotInfo, RobotCommand
from typing import Union, Dict, Optional


class AbstractRobotController:

    @abc.abstractmethod
    def send_robot_commands(self) -> None:
        """
        sends the robot commands to the appropriate team (yellow or blue).
        # TODO: Consider changing output to integer value for closed loop feedback from robots
        """
        pass

    @abc.abstractmethod
    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """
        Adds robot commands to self.out_packet.

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
            warnings.warn(
                "Invalid hyperparamters passed to add_robot_commands", SyntaxWarning
            )

    @abc.abstractmethod
    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """
        Adds a robot command to self.out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick_spd', 'kick_angle', 'dribbler_spd'.
        """
        pass
