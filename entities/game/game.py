from typing import Dict, Optional
import dataclasses
from dataclasses import dataclass, replace
from entities.game.field import Field
from entities.game.robot import Robot
from entities.game.ball import Ball
from entities.data.object import ObjectKey

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Game:
    ts: float
    my_team_is_yellow: bool
    my_team_is_right: bool
    friendly_robots: Dict[int, Robot]
    enemy_robots: Dict[int, Robot]
    ball: Optional[Ball]
    field: Field = dataclasses.field(init=False)
    robot_with_ball: Optional[ObjectKey]

    def __post_init__(self):
        object.__setattr__(self, "field", Field(self.my_team_is_right))

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
