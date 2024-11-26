from team_controller.src.controllers.common.sim_controller_abstract import (
    AbstractSimController,
)


class RSimController(AbstractSimController):
    """
    A controller for interacting with a simulation environment for robot soccer, allowing actions such as teleporting the ball
    and setting robot presence on the field.
    """

    def teleport_ball(self, x: float, y: float) -> None:
        """
        Teleports the ball to a specific location on the field.

        Args:
            x (float): The x-coordinate to place the ball at.
            y (float): The y-coordinate to place the ball at.

        This method creates a command for teleporting the ball and sends it to the simulator.
        """

    def set_robot_presence(
        self, robot_id: int, team_colour_is_yellow: bool, should_robot_be_present: bool
    ) -> None:
        """
        Sets a robot's presence on the field by teleporting it to a specific location or removing it from the field.

        Args:
            robot_id (int): The unique ID of the robot.
            team_colour_is_yellow (bool): Whether the robot belongs to the yellow team. If False, it's assumed to be blue.
            should_robot_be_present (bool): If True, the robot will be placed on the field; if False, it will be removed.

        The method calculates a teleport location based on the team and presence status, then sends a command to the simulator.
        """
