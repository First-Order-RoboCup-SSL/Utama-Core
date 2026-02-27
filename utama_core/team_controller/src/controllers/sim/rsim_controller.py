from typing import Tuple

from utama_core.config.settings import ADD_Y_COORD, REMOVAL_Y_COORD, TELEPORT_X_COORDS
from utama_core.entities.game.field import FieldBounds
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from utama_core.team_controller.src.controllers.common.sim_controller_abstract import (
    AbstractSimController,
)


class RSimController(AbstractSimController):
    """A controller for interacting with a simulation environment for robot soccer, allowing actions such as teleporting
    the ball and setting robot presence on the field."""

    def __init__(self, field_bounds: FieldBounds, env: SSLBaseEnv):
        super().__init__(field_bounds)
        self._env = env

    def _do_teleport_ball_unrestricted(self, x, y, vx, vy):
        self._env.teleport_ball(x, y, vx, vy)

    def _do_teleport_robot_unrestricted(self, is_team_yellow, robot_id, x, y, theta=None):
        self._env.teleport_robot(is_team_yellow, robot_id, x, y, theta)

    def set_robot_presence(self, robot_id: int, is_team_yellow: bool, should_robot_be_present: bool) -> None:
        """Sets a robot's presence on the field by teleporting it to a specific location or removing it from the field.

        Args:
            robot_id (int): The unique ID of the robot.
            team_colour_is_yellow (bool): Whether the robot belongs to the yellow team. If False, it's assumed to be blue.
            should_robot_be_present (bool): If True, the robot will be placed on the field; if False, it will be removed.

        The method calculates a teleport location based on the team and presence status, then sends a command to the simulator.
        """
        x, y = self._get_teleport_location(robot_id, is_team_yellow, should_robot_be_present)
        self._do_teleport_robot_unrestricted(is_team_yellow, robot_id, x, y, 0)

    def _get_teleport_location(self, robot_id: int, is_team_yellow: bool, add: bool) -> Tuple[float, float]:
        y_coord = ADD_Y_COORD if add else REMOVAL_Y_COORD
        x_coord = -TELEPORT_X_COORDS[robot_id] if not is_team_yellow else TELEPORT_X_COORDS[robot_id]
        return x_coord, y_coord

    @property
    def env(self):
        return self._env
