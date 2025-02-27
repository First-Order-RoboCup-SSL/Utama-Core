from typing import List, Optional

from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData
from entities.data.referee import RefereeData
from entities.data.command import RobotResponse

from entities.game.game_object import Colour, GameObject, Robot
from entities.game.game_object import Robot as RobotEntity
from entities.game.robot import Robot
from entities.game.ball import Ball

from entities.game.team_info import TeamInfo
from entities.referee.referee_command import RefereeCommand
from entities.referee.stage import Stage

from config.settings import TIMESTEP

import logging, warnings

logger = logging.getLogger(__name__)


class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(
        self,
        my_team_is_yellow: bool,
        my_team_is_right: bool,
        start_frame: FrameData
    ):
        self._my_team_is_yellow = my_team_is_yellow
        self._my_team_is_right = my_team_is_right
        self._field = Field(self._my_team_is_right)

        self._records: List[FrameData] = []
        self._predicted_next_frame: FrameData = None

        self._friendly_robots, self._enemy_robots = self._get_initial_robot_dicts(start_frame)       

        self._ball: Ball = Ball(start_frame.ball)

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
    def predicted_next_frame(self) -> FrameData:
        return self._predicted_next_frame

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
    def is_ball_in_goal(self, right_goal: bool):
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

    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            self._records.append(frame_data)
            self._predicted_next_frame = self._reorganise_frame(
                self.predict_frame_after(TIMESTEP)
            )
            self._update_data(frame_data)
        else:
            raise ValueError("Invalid frame data.")

    def add_robot_reponse(self, robot_responses: List[RobotResponse]) -> None:
        """Process robot reponse"""
        for robot_id, robot_info in enumerate(robot_responses):
            self._friendly_robots[robot_id].has_ball = robot_info.has_ball

    def _get_initial_robot_dicts(self, start_frame: FrameData):
        if self.my_team_is_yellow:
            friendly_robots = {id: Robot(id, is_friendly=True) for id in start_frame.yellow_robots}
            enemy_robots = {id: Robot(id, is_friendly=True) for id in start_frame.blue_robots}
        else:
            friendly_robots = {id: Robot(id, is_friendly=True) for id in start_frame.blue_robots}
            enemy_robots = {id: Robot(id, is_friendly=True) for id in start_frame.yellow_robots}

        return friendly_robots, enemy_robots

    def _update_data(self, frame_data: FrameData) -> None:
        if self.my_team_is_yellow:
            self._update_robots(frame_data.yellow_robots, frame_data.blue_robots)
        else:
            self._update_robots(frame_data.blue_robots, frame_data.yellow_robots)
        self._update_ball(frame_data.ball[0])  # Ensures BallData is correctly assigned
        
    def _update_robots(self, friendly_robot_data: List[RobotData], enemy_robot_data: List[RobotData]) -> None:
        """Updates robot data safely without exposing direct modification."""
        for robot_id, robot_data in enumerate(friendly_robot_data):
            if robot_data is not None:
                self._friendly_robots[robot_id].robot_data = robot_data

        for robot_id, robot_data in enumerate(enemy_robot_data):
            if robot_data is not None:
                self._enemy_robots[robot_id].robot_data = robot_data

    def _update_ball(self, ball_data: BallData) -> None:
        """Updates the ball's internal state instead of replacing the object."""
        if ball_data is not None:
            self._ball.ball_data = ball_data  # Ensuring we don't overwrite the Ball instance

    def get_robots_velocity(self, is_yellow: bool) -> List[tuple]:
        if len(self._records) <= 1:
            return None
        if is_yellow:
            return [
                self.get_object_velocity(RobotEntity(Colour.YELLOW, i))
                for i in range(
                    len(self.get_robots_pos(True))
                )  # TODO: This is a bit of a hack, we should be able to get the number of robots from the field
            ]
        else:
            return [
                self.get_object_velocity(RobotEntity(Colour.BLUE, i))
                for i in range(len(self.get_robots_pos(False)))
            ]

    def get_ball_pos(self) -> List[BallData]:
        if not self._records:
            return None
        warnings.warn("Use game.ball instead", DeprecationWarning, stacklevel=2)
        return self._records[-1].ball

    def get_ball_velocity(self) -> Optional[tuple]:
        return self.get_object_velocity(Ball)

    def predict_next_frame(self) -> FrameData: # TODO: Change to per object
        return self._predicted_next_frame

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
                [BallData(ball_pos[0], ball_pos[1], 0, 1)],  # TODO : Support z axis
            )

    def _reorganise_frame(self, frame: FrameData) -> Optional[FrameData]:
        if frame:
            ts, yellow_pos, blue_pos, ball_pos = frame
            if self.my_team_is_yellow:
                return FrameData(ts, yellow_pos, blue_pos, ball_pos)
            else:
                return FrameData(ts, blue_pos, yellow_pos, ball_pos)
        return None

    def _reorganise_frame_data(
        self, frame_data: FrameData, my_team_is_yellow: bool
    ) -> tuple[RobotData, RobotData, BallData]:
        _, yellow_robots, blue_robots, balls = frame_data
        if my_team_is_yellow:
            return yellow_robots, blue_robots, balls
        else:
            return blue_robots, yellow_robots, balls

    ### General Object Position Prediction ###
    def predict_object_pos_after(self, t: float, object: GameObject) -> Optional[tuple]:
        # If t is after the object has stopped we return the position at which object stopped.
        sx = 0
        sy = 0

        acceleration = self.get_object_acceleration(object)

        if acceleration is None:
            return None

        ax, ay = acceleration
        vels = self.get_object_velocity(object)

        if vels is None:
            ux, uy = None, None
        else:
            ux, uy = vels

        if object is Ball:
            ball = self.get_ball_pos()
            start_x, start_y = ball[0].x, ball[0].y
        else:
            posn = self._get_object_position_at_frame(len(self._records) - 1, object)
            start_x, start_y = posn.x, posn.y

        if ax and ux:
            sx = self._calculate_displacement(ux, ax, t)

        if ay and uy:
            sy = self._calculate_displacement(uy, ay, t)

        return (
            start_x + sx,
            start_y + sy,
        )  # TODO: Doesn't take into account spin / angular vel

    def _calculate_displacement(self, u, a, t):
        if a == 0:  # Handle zero acceleration case
            return u * t
        else:
            stop_time = -u / a
            if stop_time < 0:
                stop_time = float("inf")
            effective_time = min(t, stop_time)
            displacement = (u * effective_time) + (0.5 * a * effective_time**2)
            logger.debug(
                f"Displacement: {displacement} for time: {effective_time}, stop time: {stop_time}"
            )
            return displacement

    def predict_ball_pos_at_x(self, x: float) -> Optional[tuple]:
        vel = self.get_ball_velocity()

        if not vel or not vel[0] or not vel[0]:
            return None

        ux, uy = vel
        pos = self.get_ball_pos()[0]
        bx = pos.x
        by = pos.y

        if uy == 0:
            return (bx, by)

        t = (x - bx) / ux
        y = by + uy * t
        return (x, y)

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

        if current_pos is None or previous_pos is None:
            logger.warning("No position data to calculate velocity for frame %d", frame)
            return None

        previous_time_received = previous_frame.ts
        time_received = current_frame.ts

        if time_received < previous_time_received:
            logger.warning(
                "Timestamps out of order for vision data %f should be after %f",
                time_received,
                previous_time_received,
            )
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
            windowEnd = windowStart + WINDOW  # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                curr_vel = self._get_object_velocity_at_frame(
                    len(self._records) - j, object
                )
                if curr_vel:
                    averageVelocity[0] += curr_vel[0]
                    averageVelocity[1] += curr_vel[1]
                elif missing_velocities == WINDOW - 1:
                    logging.warning(
                        f"No velocity data to calculate acceleration for frame {len(self._records) - j}"
                    )
                    return None
                else:
                    missing_velocities += 1

            averageVelocity[0] /= WINDOW - missing_velocities
            averageVelocity[1] /= WINDOW - missing_velocities

            if i != 0:
                dt = (
                    self._records[-windowMiddle + WINDOW].ts
                    - self._records[-windowMiddle].ts
                )
                accelX = (
                    futureAverageVelocity[0] - averageVelocity[0]
                ) / dt  # TODO vec
                accelY = (futureAverageVelocity[1] - averageVelocity[1]) / dt
                totalX += accelX
                totalY += accelY
                iter += 1

            futureAverageVelocity = tuple(averageVelocity)

        return (totalX / iter, totalY / iter)

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
