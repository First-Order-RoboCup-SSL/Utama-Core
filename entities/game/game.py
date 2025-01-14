from typing import List, Optional, NamedTuple
from entities.game import game_object
from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData, PredictedFrame
from entities.data.referee import RefereeData
from entities.data.command import RobotInfo

from entities.game.game_object import Colour, GameObject, Robot
from entities.game.game_object import Robot as RobotEntity
from entities.game.robot import Robot
from entities.game.ball import Ball

from entities.game.team_info import TeamInfo
from entities.referee.referee_command import RefereeCommand
from entities.referee.stage import Stage

from team_controller.src.config.settings import TIMESTEP

import logging, warnings

logger = logging.getLogger(__name__)


class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(self, my_team_is_yellow=True):
        self._my_team_is_yellow = my_team_is_yellow
        self._field = Field()

        self._records: List[FrameData] = []
        self._predicted_next_frame: PredictedFrame = None

        self._friendly_robots: List[Robot] = [
            Robot(id, is_friendly=True) for id in range(6)
        ]
        self._enemy_robots: List[Robot] = [
            Robot(id, is_friendly=False) for id in range(6)
        ]
        self._ball: Ball = Ball()

        self._yellow_score = 0
        self._blue_score = 0
        self._referee_records = []

    @property
    def field(self) -> Field:
        return self._field

    @property
    def current_state(self) -> FrameData:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> List[FrameData]:
        return self._records if self._records else None

    @property
    def yellow_score(self) -> int:
        return self._yellow_score

    @property
    def blue_score(self) -> int:
        return self._blue_score

    @property
    def my_team_is_yellow(self) -> bool:
        return self._my_team_is_yellow

    @property
    def predicted_next_frame(self) -> PredictedFrame:
        return self._predicted_next_frame

    @property
    def friendly_robots(self) -> List[Robot]:
        return self._friendly_robots

    @friendly_robots.setter
    def friendly_robots(self, value: List[RobotData]):
        for robot_id, robot_data in enumerate(value):
            if robot_data is not None:
                self._friendly_robots[robot_id].robot_data = robot_data

    @property
    def enemy_robots(self) -> List[Robot]:
        return self._enemy_robots

    @enemy_robots.setter
    def enemy_robots(self, value: List[RobotData]):
        for robot_id, robot_data in enumerate(value):
            if robot_data is not None:
                self._enemy_robots[robot_id].robot_data = robot_data

    @property
    def ball(self) -> Ball:
        return self._ball

    @ball.setter
    def ball(self, value: BallData):
        self._ball.ball_data = value

    def is_ball_in_goal(self, left_goal: bool):
        ball_pos = self.get_ball_pos()[0]
        return (
            ball_pos.x < -self.field.HALF_LENGTH
            and (
                ball_pos.y < self.field.HALF_GOAL_WIDTH
                and ball_pos.y > -self.field.HALF_GOAL_WIDTH
            )
            and left_goal
            or ball_pos.x > self.field.HALF_LENGTH
            and (
                ball_pos.y < self.field.HALF_GOAL_WIDTH
                and ball_pos.y > -self.field.HALF_GOAL_WIDTH
            )
            and not left_goal
        )

    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            self._records.append(frame_data)
            self._predicted_next_frame = self._reorganise_frame(
                self.predict_frame_after(TIMESTEP)
            )
            self._update_data(frame_data)
        else:
            raise ValueError("Invalid frame data.")

    def add_robot_info(self, robots_info: List[RobotInfo]) -> None:
        for robot_id, robot_info in enumerate(robots_info):
            self._friendly_robots[robot_id].has_ball = robot_info.has_ball

    def _update_data(self, frame_data: FrameData) -> None:
        if self.my_team_is_yellow:
            self.friendly_robots = frame_data.yellow_robots
            self.enemy_robots = frame_data.blue_robots
        else:
            self.friendly_robots = frame_data.blue_robots
            self.enemy_robots = frame_data.yellow_robots
        self._ball = frame_data.ball[0]

    def get_robots_pos(self, is_yellow: bool) -> List[RobotData]:
        if not self._records:
            return None
        record = self._records[-1]
        warnings.warn(
            "Use game.friendly_robots/enemy_robots instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return record.yellow_robots if is_yellow else record.blue_robots

    def get_robot_pos(self, is_yellow: bool, robot_id: int) -> RobotData:
        all = self.get_robots_pos(is_yellow)
        return None if not all else all[robot_id]

    def get_robots_velocity(self, is_yellow: bool) -> List[tuple]:
        if len(self._records) <= 1:
            return None
        return [
            self.get_object_velocity(
                RobotEntity(i, Colour.YELLOW if is_yellow else Colour.BLUE)
            )
            for i in range(len(self.get_robots_pos(is_yellow)))
        ]

    def get_ball_pos(self) -> List[BallData]:
        if not self._records:
            return None
        warnings.warn("Use game.ball instead", DeprecationWarning, stacklevel=2)
        return self._records[-1].ball

    def get_ball_velocity(self) -> Optional[tuple]:
        return self.get_object_velocity(Ball)

    def get_latest_frame(self) -> Optional[FrameData]:
        return self._records[-1] if self._records else None

    def get_my_latest_frame(
        self, my_team_is_yellow: bool
    ) -> tuple[RobotData, RobotData, BallData]:
        if not self._records:
            return None
        latest_frame = self.get_latest_frame()
        warnings.warn(
            "Use game.friendly/enemy.x/y/orentation instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._reorganise_frame_data(latest_frame, my_team_is_yellow)

    def predict_next_frame(self) -> FrameData:
        return self._predicted_next_frame

    def predict_my_next_frame(
        self, my_team_is_yellow: bool
    ) -> tuple[RobotData, RobotData, BallData]:
        if self._predicted_next_frame is None:
            return None
        warnings.warn(
            "Use game.predicted_next_frame instead", DeprecationWarning, stacklevel=2
        )
        return self._reorganise_frame_data(
            self._predicted_next_frame, my_team_is_yellow
        )

    def predict_frame_after(self, t: float) -> FrameData:
        yellow_pos = [
            self.predict_object_pos_after(t, RobotEntity(Colour.YELLOW, i))
            for i in range(len(self.get_robots_pos(True)))
        ]
        blue_pos = [
            self.predict_object_pos_after(t, RobotEntity(Colour.BLUE, i))
            for i in range(len(self.get_robots_pos(False)))
        ]
        ball_pos = self.predict_object_pos_after(t, Ball)
        if ball_pos is None or None in yellow_pos or None in blue_pos:
            return None
        else:
            return FrameData(
                self._records[-1].ts + t,
                list(map(lambda pos: RobotData(pos[0], pos[1], 0), yellow_pos)),
                list(map(lambda pos: RobotData(pos[0], pos[1], 0), blue_pos)),
                [BallData(ball_pos[0], ball_pos[1], 0)],
            )

    def _reorganise_frame(self, frame: FrameData) -> Optional[PredictedFrame]:
        if frame:
            ts, yellow_pos, blue_pos, ball_pos = frame
            if self.my_team_is_yellow:
                return PredictedFrame(ts, yellow_pos, blue_pos, ball_pos)
            else:
                return PredictedFrame(ts, blue_pos, yellow_pos, ball_pos)
        return None

    def _reorganise_frame_data(
        self, frame_data: FrameData, my_team_is_yellow: bool
    ) -> tuple[RobotData, RobotData, BallData]:
        _, yellow_robots, blue_robots, balls = frame_data
        return (
            (yellow_robots, blue_robots, balls)
            if my_team_is_yellow
            else (blue_robots, yellow_robots, balls)
        )

    def get_object_velocity(self, object: GameObject) -> Optional[tuple]:
        return self._get_object_velocity_at_frame(len(self._records) - 1, object)

    def _get_object_position_at_frame(self, frame: int, object: GameObject):
        if object == Ball:
            return self._records[frame].ball[0]
        elif isinstance(object, RobotEntity):
            return (
                self._records[frame].yellow_robots[object.id]
                if object.colour == Colour.YELLOW
                else self._records[frame].blue_robots[object.id]
            )

    def _get_object_velocity_at_frame(
        self, frame: int, object: GameObject
    ) -> Optional[tuple]:
        if frame >= len(self._records) or frame == 0:
            logger.warning("Cannot provide velocity at a frame that does not exist")
            return None

        previous_frame = self._records[frame - 1]
        current_frame = self._records[frame]

        previous_pos = self._get_object_position_at_frame(frame - 1, object)
        current_pos = self._get_object_position_at_frame(frame, object)

        previous_time_received = previous_frame.ts
        time_received = current_frame.ts

        if time_received < previous_time_received:
            logger.warning("Timestamps out of order for vision data")
            return None

        dt_secs = time_received - previous_time_received

        vx = (current_pos.x - previous_pos.x) / dt_secs
        vy = (current_pos.y - previous_pos.y) / dt_secs

        return (vx, vy)

    def get_object_acceleration(self, object: GameObject) -> Optional[tuple]:
        totalX = 0
        totalY = 0
        WINDOW = 5
        N_WINDOWS = 3
        iter = 0
        missing_velocities = 0

        if len(self._records) < WINDOW * N_WINDOWS + 1:
            return None

        for i in range(N_WINDOWS):
            missing_velocities = 0
            averageVelocity = [0, 0]
            windowStart = 1 + (i * WINDOW)
            windowEnd = windowStart + WINDOW
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                curr_vel = self._get_object_velocity_at_frame(
                    len(self._records) - j, object
                )
                if curr_vel:
                    averageVelocity[0] += curr_vel[0]
                    averageVelocity[1] += curr_vel[1]
                else:
                    missing_velocities += 1

            averageVelocity[0] /= WINDOW - missing_velocities
            averageVelocity[1] /= WINDOW - missing_velocities

            if i != 0:
                dt = (
                    self._records[-windowMiddle + WINDOW].ts
                    - self._records[-windowMiddle].ts
                )
                accelX = (futureAverageVelocity[0] - averageVelocity[0]) / dt
                accelY = (futureAverageVelocity[1] - averageVelocity[1]) / dt
                totalX += accelX
                totalY += accelY
                iter += 1

            futureAverageVelocity = tuple(averageVelocity)

        return (totalX / iter, totalY / iter)

    def predict_object_pos_after(self, t: float, object: GameObject) -> Optional[tuple]:
        acc = self.get_object_acceleration(object)
        if acc is None:
            return None

        ax, ay = acc
        ux, uy = self.get_object_velocity(object)

        if object is Ball:
            ball = self.get_ball_pos()
            start_x, start_y = ball[0].x, ball[0].y
        else:
            posn = self._get_object_position_at_frame(len(self._records) - 1, object)
            start_x, start_y = posn.x, posn.y

        if ax == 0:
            sx = 0
        else:
            tx_stop = -ux / ax
            tx = min(t, tx_stop)
            sx = ux * tx + 0.5 * ax * tx * tx

        if ay == 0:
            sy = 0
        else:
            ty_stop = -uy / ay
            ty = min(t, ty_stop)
            sy = uy * ty + 0.5 * ay * ty * ty

        return (start_x + sx, start_y + sy)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    game = Game()
    print(game.ball.x)
    print(game.ball.y)
    print(game.ball.z)
    game.ball = BallData(1, 2, 3)
    print(game.ball.x)
    print(game.ball.y)
    print(game.ball.z)
