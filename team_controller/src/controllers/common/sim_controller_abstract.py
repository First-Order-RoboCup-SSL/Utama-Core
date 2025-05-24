import abc


class AbstractSimController:
    """
    Template for generic sim controller, allowing actions such
    """

    @abc.abstractmethod
    def teleport_ball(self, x: float, y: float) -> None:
        """
        Teleports the ball to a specific location on the field.

        Args:
            x (float): The x-coordinate to place the ball at (in meters [-4.5, 4.5]).
            y (float): The y-coordinate to place the ball at (in meters [-3.0, 3.0]).

        This method creates a command for teleporting the ball and sends it to the simulator.
        """
        ...

    @abc.abstractmethod
    def teleport_robot(
        self,
        is_team_yellow: bool,
        robot_id: int,
        x: float,
        y: float,
        theta: float = None,
    ) -> None:
        """
        Teleports a robot to a specific location on the field.

        Args:
            is_team_yellow (bool): if the robot is team yellow, else blue
            robot_id (int): robot id
            x (float): The x-coordinate to place the ball at (in meters [-4.5, 4.5]).
            y (float): The y-coordinate to place the ball at (in meters [-3.0, 3.0]).
            theta (float): radian angle of the robot heading, 0 degrees faces towards positive x axis

        This method creates a command for teleporting the ball and sends it to the simulator.
        """
        ...

    @abc.abstractmethod
    def set_robot_presence(
        self, is_team_yellow: bool, robot_id: int, should_robot_be_present: bool
    ) -> None:
        """
        Sets a robot's presence on the field by teleporting it to a specific location or removing it from the field.

        Args:
            robot_id (int): The unique ID of the robot.
            team_colour_is_blue (bool): Whether the robot belongs to the blue team. If False, it's assumed to be yellow.
            should_robot_be_present (bool): If True, the robot will be placed on the field; if False, it will be removed.

        The method calculates a teleport location based on the team and presence status, then sends a command to the simulator.
        """
        ...
