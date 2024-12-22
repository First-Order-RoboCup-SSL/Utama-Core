from typing import Tuple, Optional, List, Union
from dataclasses import field
import numpy as np

from entities.game.role import Role, RoleType
from team_controller.src.controllers import GRSimController
from team_controller.src.controllers import RSimController


class Robot:
    def _init_(self, robot_id, is_team_yellow):
        self._robot_id: int = robot_id
        self._is_team_yellow: bool = is_team_yellow
        self._role: Role = Role(self.robot_id)
        self._has_ball: bool = False
        self._pos: Optional[Tuple[float, float]] = None
        self._heading: Optional[float] = None
        self._aggro_rating: int = 0
        self._records: List[str] = field(default_factory=list)
        self._controller: Union[GRSimController, RSimController] = (
            None  # TODO: is this needed? we should be doing robot manipulation from team_controller
            # consider removing this and writing a function for updating robot state (ie updating based on what is observed from the game)
        )

    def change_role(self, role: Union[RoleType, str]) -> None:
        self._role.change_role(role)

    def set_position(self, x: float, y: float) -> None:
        if x < 0 or y < 0:
            raise ValueError("Position coordinates must be non-negative")
        self._pos = (x, y)
        self._controller.teleport_robot(
            is_team_yellow=self.is_team_yellow, robot_id=self.robot_id, x=x, y=y
        )

    def set_theta(self, theta: float) -> None:
        if not (-np.pi < theta <= np.pi):
            raise ValueError("Heading must be in range (-pi, pi]")
        self._heading = theta
        self.sim_controller.teleport_robot(
            is_team_yellow=self.is_team_yellow, robot_id=self.robot_id, theta=self.theta
        )

    def update_has_ball(self, has_ball: bool) -> None:
        self._has_ball = has_ball

    def update_agro_rating(self, aggro_rating: float):
        if aggro_rating >= 0:
            self._aggro_rating = aggro_rating
        else:
            raise ValueError("agro raiting value must be non-negative")

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def is_team_yellow(self):
        return self._is_team_yellow

    @property
    def role(self):
        return self._role

    @property
    def has_ball(self):
        return self._has_ball

    @property
    def pos(self):
        return self._pos

    @property
    def heading(self):
        return self._heading

    @property
    def aggro_rating(self):
        return self._aggro_rating

    @property
    def records(self):
        return self._records

    @property
    def controller(self):
        return self._controller
