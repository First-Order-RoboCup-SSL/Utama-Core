from typing import Dict
import dataclasses
from dataclasses import dataclass, replace

from entities.game.robot import Robot
from entities.game.ball import Ball

import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Game:
    ts: float
    my_team_is_yellow: bool
    my_team_is_right: bool
    field: float = dataclasses.field(init=False)
    friendly_robots: Dict[int, Robot] = dataclasses.field(default={}, init=False)
    enemy_robots: Dict[int, Robot] = dataclasses.field(default={}, init=False)
    ball: Ball = dataclasses.field(default=None, init=False)

    def is_ball_in_goal(self, right_goal: bool) -> bool:
        ball_pos = self.ball
        return (
            ball_pos.x < -self.field.half_length
            and (
                ball_pos.y < self.field.half_goal_width
                and ball_pos.y > -self.field.half_goal_width
            )
            and not right_goal
            or ball_pos.x > self.field.half_length
            and (
                ball_pos.y < self.field.half_goal_width
                and ball_pos.y > -self.field.half_goal_width
            )
            and right_goal
        )
