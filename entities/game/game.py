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


# Only to be used in this file
def combine_robot_vision_data(old_robot: Robot, robot_data: VisionRobotData) -> Robot:
    assert old_robot.id == robot_data.id
    return replace(old_robot,
        id=robot_data.id,
        x=robot_data.x,
        y=robot_data.y,
        orientation=robot_data.orientation,
    )


# Used at start of the game so assume robot does not have the ball
def robot_from_vision(robot_data: VisionRobotData, is_friendly: bool) -> Robot:
    return Robot(
        id=robot_data.id,
        is_friendly=is_friendly,
        has_ball=False,
        x=robot_data.x,
        y=robot_data.y,
        orientation=robot_data.orientation,
    )


def ball_from_vision(ball_data: VisionBallData) -> Ball:
    return Ball(ball_data.x, ball_data.y, ball_data.z)


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

        assert len(start_frame.ball) > 0, (
            "No ball data in start frame, must give ball start position"
        )
        assert len(start_frame.yellow_robots) > 0, (
            "No yellow robot data in start frame, must give yellow robot start positions"
        )
        assert len(start_frame.blue_robots) > 0, (
            "No blue robot data in start frame, must give blue robot start positions"
        )

        self._friendly_robots, self._enemy_robots = self._get_initial_robot_dicts(
            start_frame
        )

        self._ball: Ball = self._get_most_confident_ball(start_frame.ball)

        self._referee_records = []

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

    # Put in the field class?
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

    def _get_initial_robot_dicts(
        self, start_frame: VisionData
    ) -> Tuple[Dict[int, Robot], Dict[int, Robot]]:
        if self.my_team_is_yellow:
            friendly_robots = {
                rd.id: robot_from_vision(rd, is_friendly=True)
                for rd in start_frame.yellow_robots
            }
            enemy_robots = {
                rd.id: robot_from_vision(rd, is_friendly=False)
                for rd in start_frame.blue_robots
            }
        else:
            friendly_robots = {
                rd.id: robot_from_vision(rd, is_friendly=True)
                for rd in start_frame.blue_robots
            }
            enemy_robots = {
                rd.id: robot_from_vision(rd, is_friendly=False)
                for rd in start_frame.yellow_robots
            }

        return friendly_robots, enemy_robots

    def _get_most_confident_ball(self, balls: List[VisionBallData]) -> Ball:
        balls_by_confidence = sorted(
            balls, key=lambda ball: ball.confidence, reverse=True
        )
        return Ball(
            balls_by_confidence[0].x, balls_by_confidence[0].y, balls_by_confidence[0].z
        )

    def _update_data(self, frame_data: VisionData) -> None:
        if self.my_team_is_yellow:
            self._update_robots(frame_data.yellow_robots, frame_data.blue_robots)
        else:
            self._update_robots(frame_data.blue_robots, frame_data.yellow_robots)
        self._update_balls(frame_data.ball)

    def _update_robots(
        self, friendly_robot_data: List[VisionRobotData], enemy_robot_data: List[VisionRobotData]
    ) -> None:
        for robot_data in friendly_robot_data:
            self._friendly_robots[robot_data.id] = combine_robot_vision_data(
                self._friendly_robots[robot_data.id], robot_data
            )

        for robot_data in enemy_robot_data:
            self._enemy_robots[robot_data.id] = combine_robot_vision_data(
                self._enemy_robots[robot_data.id], robot_data
            )

    def _update_balls(self, balls_data: List[VisionBallData]) -> None:
        # Does not update when there is nothing to update
        if balls_data:
            self._ball = ball_from_vision(self._get_most_confident_ball(balls_data))

    def add_new_referee_data(self, referee_data: RefereeData) -> None:
        if not self._referee_records:
            self._referee_records.append(referee_data)
        elif referee_data[1:] != self._referee_records[-1][1:]:
            self._referee_records.append(referee_data)

    def source_identifier(self) -> Optional[str]:
        return (
            self._referee_records[-1].source_identifier
            if self._referee_records
            else None
        )

    @property
    def last_time_sent(self) -> float:
        return self._referee_records[-1].time_sent if self._referee_records else 0.0

    @property
    def last_time_received(self) -> float:
        return self._referee_records[-1].time_received if self._referee_records else 0.0

    @property
    def last_command(self) -> RefereeCommand:
        return (
            self._referee_records[-1].referee_command
            if self._referee_records
            else RefereeCommand.HALT
        )

    @property
    def last_command_timestamp(self) -> float:
        return (
            self._referee_records[-1].referee_command_timestamp
            if self._referee_records
            else 0.0
        )

    @property
    def stage(self) -> Stage:
        return (
            self._referee_records[-1].stage
            if self._referee_records
            else Stage.NORMAL_FIRST_HALF_PRE
        )

    @property
    def stage_time_left(self) -> float:
        return (
            self._referee_records[-1].stage_time_left if self._referee_records else 0.0
        )

    @property
    def blue_team(self) -> TeamInfo:
        return (
            self._referee_records[-1].blue_team
            if self._referee_records
            else TeamInfo(
                name="",
                score=0,
                red_cards=0,
                yellow_card_times=[],
                yellow_cards=0,
                timeouts=0,
                timeout_time=0,
                goalkeeper=0,
            )
        )

    @property
    def yellow_team(self) -> TeamInfo:
        return (
            self._referee_records[-1].yellow_team
            if self._referee_records
            else TeamInfo(
                name="",
                score=0,
                red_cards=0,
                yellow_card_times=[],
                yellow_cards=0,
                timeouts=0,
                timeout_time=0,
                goalkeeper=0,
            )
        )

    @property
    def designated_position(self) -> Optional[tuple[float]]:
        return (
            self._referee_records[-1].designated_position
            if self._referee_records
            else None
        )

    @property
    def blue_team_on_positive_half(self) -> Optional[bool]:
        return (
            self._referee_records[-1].blue_team_on_positive_half
            if self._referee_records
            else None
        )

    @property
    def next_command(self) -> Optional[RefereeCommand]:
        return self._referee_records[-1].next_command if self._referee_records else None

    @property
    def current_action_time_remaining(self) -> Optional[int]:
        return (
            self._referee_records[-1].current_action_time_remaining
            if self._referee_records
            else None
        )

    @property
    def is_halt(self) -> bool:
        return self.last_command == RefereeCommand.HALT

    @property
    def is_stop(self) -> bool:
        return self.last_command == RefereeCommand.STOP

    @property
    def is_normal_start(self) -> bool:
        return self.last_command == RefereeCommand.NORMAL_START

    @property
    def is_force_start(self) -> bool:
        return self.last_command == RefereeCommand.FORCE_START

    @property
    def is_prepare_kickoff_yellow(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    @property
    def is_prepare_kickoff_blue(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_KICKOFF_BLUE

    @property
    def is_prepare_penalty_yellow(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_PENALTY_YELLOW

    @property
    def is_prepare_penalty_blue(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_PENALTY_BLUE

    @property
    def is_direct_free_yellow(self) -> bool:
        return self.last_command == RefereeCommand.DIRECT_FREE_YELLOW

    @property
    def is_direct_free_blue(self) -> bool:
        return self.last_command == RefereeCommand.DIRECT_FREE_BLUE

    @property
    def is_timeout_yellow(self) -> bool:
        return self.last_command == RefereeCommand.TIMEOUT_YELLOW

    @property
    def is_timeout_blue(self) -> bool:
        return self.last_command == RefereeCommand.TIMEOUT_BLUE

    @property
    def is_ball_placement_yellow(self) -> bool:
        return self.last_command == RefereeCommand.BALL_PLACEMENT_YELLOW

    @property
    def is_ball_placement_blue(self) -> bool:
        return self.last_command == RefereeCommand.BALL_PLACEMENT_BLUE


