from typing import List, Optional, Tuple, Dict
from dataclasses import replace

from entities.game.field import Field
from entities.data.vision import VisionData, VisionRobotData, VisionBallData
from entities.data.referee import RefereeData

from entities.game.robot import Robot
from entities.game.ball import Ball

from entities.game.team_info import TeamInfo
from entities.referee.referee_command import RefereeCommand
from entities.referee.stage import Stage

import logging

logger = logging.getLogger(__name__)

class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(
        self, my_team_is_yellow: bool, my_team_is_right: bool
    ):
        self._my_team_is_yellow = my_team_is_yellow
        self._my_team_is_right = my_team_is_right
        self._field = Field(self._my_team_is_right)

        self._records: List[VisionData] = []

        self._friendly_robots, self._enemy_robots = {}, {}

        self._ball: Ball = None

    @property
    def field(self) -> Field:
        return self._field

    @property
    def my_team_is_yellow(self) -> bool:
        return self._my_team_is_yellow

    @property
    def friendly_robots(self) -> List[Robot]:
        return self._friendly_robots

    @property
    def enemy_robots(self) -> List[Robot]:
        return self._enemy_robots

    @property
    def ball(self) -> Ball:
        return self._ball

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

    def add_new_state(self, frame_data: VisionData) -> None:
        if isinstance(frame_data, VisionData):
            self._records.append(frame_data)
            self._update_data(frame_data)
        else:
            raise ValueError("Invalid frame data.")

