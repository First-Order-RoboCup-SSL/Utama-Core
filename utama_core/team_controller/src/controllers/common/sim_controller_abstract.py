import abc

from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import in_field_bounds


class AbstractSimController:
    """Template for generic sim controller, allowing actions such."""

    def __init__(self, field_bounds: FieldBounds, exp_ball: bool = True):
        self.field_bounds = field_bounds
        self.exp_ball = exp_ball

    def remove_ball(self):
        """Removes the ball from the field by teleporting it two times the field dimensions away."""
        self._do_teleport_ball_unrestricted(
            self.field_bounds.bottom_right[0] * 2,
            self.field_bounds.bottom_right[1] * 2,
            0,
            0,
        )

    def teleport_ball(self, x: float, y: float, vx: float = 0, vy: float = 0) -> None:
        """Teleports the ball to a specific location on the field.

        Args:
            x (float): The x-coordinate to place the ball in meters within FieldBounds.
            y (float): The y-coordinate to place the ball in meters within FieldBounds.
            vx (float): The velocity of the ball in the x-direction (in meters per second).
            vy (float): The velocity of the ball in the y-direction (in meters per second).

        Method does not allow for teleporting the ball outside of the field boundaries.

        This method creates a command for teleporting the ball and sends it to the simulator.
        """
        if self.exp_ball is False:
            raise ValueError(
                "This controller is configured to not expect a ball, so teleporting the ball is not allowed."
            )
        in_field = in_field_bounds((x, y), self.field_bounds)
        if not in_field:
            raise ValueError(
                f"Cannot teleport ball to ({x}, {y}) as it is outside of the field boundaries defined by {self.field_bounds}."
            )
        self._do_teleport_ball_unrestricted(x, y, vx, vy)

    def teleport_robot(
        self,
        is_team_yellow: bool,
        robot_id: int,
        x: float,
        y: float,
        theta: float = None,
    ) -> None:
        """Teleports a robot to a specific location on the field.

        Args:
            is_team_yellow (bool): if the robot is team yellow, else blue
            robot_id (int): robot id
            x (float): The x-coordinate to place the robot in meters within FieldBounds.
            y (float): The y-coordinate to place the robot in meters within FieldBounds.
            theta (float): radian angle of the robot heading, 0 degrees faces towards positive x axis

        Method does not allow for teleporting the robot outside of the field boundaries.

        This method creates a command for teleporting the robot and sends it to the simulator.
        """
        in_field = in_field_bounds((x, y), self.field_bounds)
        if not in_field:
            raise ValueError(
                f"Cannot teleport robot to ({x}, {y}) as it is outside of the field boundaries defined by {self.field_bounds}."
            )
        self._do_teleport_robot_unrestricted(is_team_yellow, robot_id, x, y, theta)

    ### Below methods are implemented in the specific sim controllers ####

    @abc.abstractmethod
    def set_robot_presence(self, robot_id: int, is_team_yellow: bool, should_robot_be_present: bool) -> None:
        """Sets a robot's presence on the field by teleporting it to a specific location or removing it from the field.

        Args:
            robot_id (int): The unique ID of the robot.
            is_team_yellow (bool): Whether the robot belongs to the yellow team. If False, it's assumed to be blue.
            should_robot_be_present (bool): If True, the robot will be placed on the field; if False, it will be removed.

        The method calculates a teleport location based on the team and presence status, then sends a command to the simulator.
        """
        ...

    @abc.abstractmethod
    def _do_teleport_ball_unrestricted(self, x: float, y: float, vx: float, vy: float) -> None:
        """Teleports the ball to a specific location on the field without any boundary checks."""
        ...

    @abc.abstractmethod
    def _do_teleport_robot_unrestricted(
        self,
        is_team_yellow: bool,
        robot_id: int,
        x: float,
        y: float,
        theta: float = None,
    ) -> None:
        """Teleports a robot to a specific location on the field without any boundary checks."""
        ...

    ### End of abstract methods ###
