import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field
from utama_core.entities.game.robot import Robot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameFrame:
    ts: float
    my_team_is_yellow: bool
    my_team_is_right: bool
    friendly_robots: Dict[int, Robot]
    enemy_robots: Dict[int, Robot]
    ball: Optional[Ball]
    referee: Optional[RefereeData] = None

    def is_ball_in_goal(self, right_goal: bool) -> bool:
        ball_pos = self.ball.p
        return (
            ball_pos.x < -self.field.half_length
            and (ball_pos.y < self.field.half_goal_width and ball_pos.y > -self.field.half_goal_width)
            and not right_goal
            or ball_pos.x > self.field.half_length
            and (ball_pos.y < self.field.half_goal_width and ball_pos.y > -self.field.half_goal_width)
            and right_goal
        )
