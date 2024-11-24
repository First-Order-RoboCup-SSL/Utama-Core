import abc
from entities.data.command import RobotInfo, RobotSimCommand
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
        robot_commands: Union[RobotSimCommand, Dict[int, RobotSimCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """
        Adds robot commands to the out_packet.

        Args:
            robot_commands (Union[RobotSimCommand, Dict[int, RobotSimCommand]]): A single RobotSimCommand or a dictionary of RobotSimCommand with robot_id as the key.
            robot_id (Optional[int]): The ID of the robot which is ONLY used when adding one Robot command. Defaults to None.

        Raises:
            SyntaxWarning: If invalid hyperparameters are passed to the function.
        """
        pass
