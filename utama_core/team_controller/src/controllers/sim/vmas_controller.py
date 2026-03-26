"""VmasController: Mirrors RSimController for the VMAS backend."""

from typing import Tuple

from utama_core.config.settings import ADD_Y_COORD, REMOVAL_Y_COORD, TELEPORT_X_COORDS
from utama_core.entities.game.field import FieldBounds
from utama_core.team_controller.src.controllers.common.sim_controller_abstract import (
    AbstractSimController,
)
from utama_core.vmas_simulator.src.ssl.ssl_vmas_base import SSLVmasBaseEnv


class VmasController(AbstractSimController):
    """A controller for interacting with the VMAS simulation environment,
    allowing teleportation and robot presence management.

    Same interface as RSimController. Uses SSL standard coordinates.
    """

    def __init__(self, field_bounds: FieldBounds, exp_ball: bool, env: SSLVmasBaseEnv):
        super().__init__(field_bounds, exp_ball)
        self._env = env

    def _do_teleport_ball_unrestricted(self, x: float, y: float, vx: float, vy: float) -> None:
        self._env.teleport_ball(x, y, vx, vy)

    def _do_teleport_robot_unrestricted(
        self,
        is_team_yellow: bool,
        robot_id: int,
        x: float,
        y: float,
        theta: float = None,
    ):
        self._env.teleport_robot(is_team_yellow, robot_id, x, y, theta)

    def set_robot_presence(self, robot_id: int, is_team_yellow: bool, should_robot_be_present: bool) -> None:
        x, y = self._get_teleport_location(robot_id, is_team_yellow, should_robot_be_present)
        self.teleport_robot(is_team_yellow, robot_id, x, y, 0)

    def _get_teleport_location(self, robot_id: int, is_team_yellow: bool, add: bool) -> Tuple[float, float]:
        y_coord = ADD_Y_COORD if add else REMOVAL_Y_COORD
        x_coord = -TELEPORT_X_COORDS[robot_id] if not is_team_yellow else TELEPORT_X_COORDS[robot_id]
        return x_coord, y_coord

    @property
    def env(self):
        return self._env
