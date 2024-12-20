from typing import Tuple, Optional, List, Union
from dataclasses import dataclass, field
import numpy as np

from entities.game.role import Role
from team_controller.src.controllers import GRSimController
from team_controller.src.controllers import RSimController

ROLES = ["attack", "defend", "keeper", "ball_placement"]

@dataclass
class Robot:
    robot_id: int
    is_team_yellow: bool
    role: Role = field(init=False)
    has_ball: bool = False
    pos: Optional[Tuple[float, float]] = None
    theta: Optional[float] = None
    agro_rating: int = 0
    records: List[str] = field(default_factory=list)
    using_grsim: bool = True
    sim_controller = GRSimController if using_grsim else RSimController

    def __post_init__(self):
        self.role = Role(self.robot_id)

    def change_role(self, role: Union[int, str]) -> None:
        self.role.change_role(role)

    def set_position(self, x: float, y: float) -> None:
        if x < 0 or y < 0:
            raise ValueError("Position coordinates must be non-negative")
        self.pos = (x, y)
        self.sim_controller.teleport_robot(
        is_team_yellow=self.is_team_yellow, robot_id=self.robot_id, x=x, y=y
        )

    def set_theta(self, theta: float) -> None:
        if not (-np.pi < theta <= np.pi):
            raise ValueError("Heading must be in range (-pi, pi]")
        self.heading = theta
        self.sim_controller.teleport_robot(
        is_team_yellow=self.is_team_yellow, robot_id=self.robot_id,theta=self.theta
        )

    def update_has_ball(self, has_ball: bool) -> None:
        self.has_ball = has_ball
        
    def update_agro_rating(self, agro_rating: float):
        if agro_rating >= 0 :
            self.agro_rating = agro_rating
        else:
            raise ValueError("agro raiting value must be non-negative")
