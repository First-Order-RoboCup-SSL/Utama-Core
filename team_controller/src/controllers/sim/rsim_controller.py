from team_controller.src.controllers.common.sim_controller_abstract import (
    AbstractSimController,
)
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from team_controller.src.config.settings import (
    LOCAL_HOST,
    SIM_CONTROL_PORT,
    TELEPORT_X_COORDS,
    ADD_Y_COORD,
    REMOVAL_Y_COORD,
)
from typing import Tuple


class RSimController(AbstractSimController):
    """
    A controller for interacting with a simulation environment for robot soccer, allowing actions such as teleporting the ball
    and setting robot presence on the field.
    """

    def __init__(self, env: SSLBaseEnv):
        self._env = env

    def teleport_ball(self, x: float, y: float, vx: float = 0, vy: float = 0) -> None:
        """
        Teleports the ball to a specific location on the field.

        Args:
            x (float): The x-coordinate to place the ball at (in meters [-4.5, 4.5]).
            y (float): The y-coordinate to place the ball at (in meters [-3.0, 3.0]).

        This method creates a command for teleporting the ball and sends it to the simulator.
        """
        self._env.teleport_ball(x, y, vx, vy)

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
        self._env.teleport_robot(is_team_yellow, robot_id, x, y, theta)

    def set_robot_presence(
        self, robot_id: int, is_team_yellow: bool, should_robot_be_present: bool
    ) -> None:
        """
        Sets a robot's presence on the field by teleporting it to a specific location or removing it from the field.

        Args:
            robot_id (int): The unique ID of the robot.
            team_colour_is_yellow (bool): Whether the robot belongs to the yellow team. If False, it's assumed to be blue.
            should_robot_be_present (bool): If True, the robot will be placed on the field; if False, it will be removed.

        The method calculates a teleport location based on the team and presence status, then sends a command to the simulator.
        """
        x, y = self._get_teleport_location(
            robot_id, is_team_yellow, should_robot_be_present
        )
        self._env.teleport_robot(is_team_yellow, robot_id, x, y)

    def _get_teleport_location(
        self, robot_id: int, is_team_yellow: bool, add: bool
    ) -> Tuple[float, float]:
        y_coord = ADD_Y_COORD if add else REMOVAL_Y_COORD
        x_coord = (
            -TELEPORT_X_COORDS[robot_id]
            if not is_team_yellow
            else TELEPORT_X_COORDS[robot_id]
        )
        return x_coord, y_coord

    @property
    def env(self):
        return self._env
