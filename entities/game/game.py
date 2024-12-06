from typing import List, Optional, NamedTuple
from entities.game import game_object
from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData, PredictedFrame
from entities.data.referee import RefereeData
<<<<<<< HEAD
from entities.data.command import RobotInfo

from entities.game.game_object import Colour, GameObject, Robot
from entities.game.game_object import Robot as RobotEntity
from entities.game.robot import Robot
from entities.game.ball import Ball

from entities.game.team_info import TeamInfo
from entities.referee.referee_command import RefereeCommand
=======
from entities.game.game_object import Ball, Colour, GameObject, Robot
from entities.game.team_info import TeamInfo
from entities.referee.referee_command import RefereeCommand
from entities.referee.stage import Stage

>>>>>>> 31d8bcc (Change RefereeCommand and Stage to enum class)
from team_controller.src.config.settings import TIMESTEP

# TODO : ^ I don't like this circular import logic. Wondering if we should store this constant somewhere else

import logging, warnings

# Configure logging
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
        if not self._records:
            return None
        return self._records

    @property
    def yellow_score(self) -> int:
        return (
            self._yellow_score
        )  # TODO, read directly from _referee_records or store as class variable?

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
            # TODO: temporary fix for robot data being None
            if robot_data is not None:
                self._friendly_robots[robot_id].robot_data = robot_data

    @property
    def enemy_robots(self) -> List[Robot]:
        return self._enemy_robots

    @enemy_robots.setter
    def enemy_robots(self, value: List[RobotData]):
        for robot_id, robot_data in enumerate(value):
            # TODO: temporary fix for robot data being None
            if robot_data is not None:
                self._enemy_robots[robot_id].robot_data = robot_data

    @property
    def ball(self) -> Ball:
        return self._ball

    @ball.setter
    # TODO: can always make a "setter" which copies the object and returns a new object with the changed value
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

    ### Game state management ###
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
            # Extensible with more info (remeber to add the property in robot.py)

    def _update_data(self, frame_data: FrameData) -> None:
        if self.my_team_is_yellow:
            self.friendly_robots = frame_data.yellow_robots
            self.enemy_robots = frame_data.blue_robots
        else:
            self.friendly_robots = frame_data.blue_robots
            self.enemy_robots = frame_data.yellow_robots
        self._ball = frame_data.ball[0]  # TODO: Don't always take first ball pos

    ### Robot data retrieval ###
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
        """
        Returns (vx, vy) of all robots on a team at the latest frame. None if no data available.
        """
        if len(self._records) <= 1:
            return None
        if is_yellow:
            # TODO: potential namespace conflict when robot (robot.py) entity is reintroduced. Think about integrating the two
            return [
                self.get_object_velocity(RobotEntity(i, Colour.YELLOW))
                for i in range(
                    len(self.get_robots_pos(True))
                )  # TODO: This is a bit of a hack, we should be able to get the number of robots from the field
            ]
        else:
            return [
                self.get_object_velocity(RobotEntity(i, Colour.BLUE))
                for i in range(len(self.get_robots_pos(False)))
            ]

    ### Ball Data retrieval ###
    def get_ball_pos(self) -> List[BallData]:
        if not self._records:
            return None
        warnings.warn("Use game.ball instead", DeprecationWarning, stacklevel=2)
        return self._records[-1].ball

    def get_ball_velocity(self) -> Optional[tuple]:
        """
        Returns (vx, vy) of the ball at the latest frame. None if no data available.
        """
        return self.get_object_velocity(Ball)

    ### Frame Data retrieval ###
    def get_latest_frame(self) -> Optional[FrameData]:
        if not self._records:
            return None
        return self._records[-1]

    def get_my_latest_frame(
        self, my_team_is_yellow: bool
    ) -> tuple[RobotData, RobotData, BallData]:
        """
        FrameData rearranged as Tuple(friendly_robots, enemy_robots, balls) based on provided _my_team_is_yellow field
        """
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
        """
        Predicts the next frame based on the latest frame.
        """
        return self._predicted_next_frame

    def predict_my_next_frame(
        self, my_team_is_yellow: bool
    ) -> tuple[RobotData, RobotData, BallData]:
        """
        FrameData rearranged as (friendly_robots, enemy_robots, balls) based on my_team_is_yellow
        """
        if self._predicted_next_frame is None:
            return None
        warnings.warn(
            "Use game.predicted_next_frame instead", DeprecationWarning, stacklevel=2
        )
        return self._reorganise_frame_data(self._predicted_next_frame)

    def predict_frame_after(self, t: float) -> FrameData:
        """
        Predicts frame in t seconds from the latest frame.
        """
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
                [BallData(ball_pos[0], ball_pos[1], 0)],  # TODO : Support z axis
            )

    def _reorganise_frame(self, frame: FrameData) -> Optional[PredictedFrame]:
        if frame:
            ts, yellow_pos, blue_pos, ball_pos = frame
            if self.my_team_is_yellow:
                return PredictedFrame(
                    ts,
                    yellow_pos,
                    blue_pos,
                    ball_pos,
                )
            else:
                return PredictedFrame(
                    ts,
                    blue_pos,
                    yellow_pos,
                    ball_pos,
                )
        return None

    def _reorganise_frame_data(
        self, frame_data: FrameData, my_team_is_yellow: bool
    ) -> tuple[RobotData, RobotData, BallData]:
        """
        *Deprecated* reorganises frame data to be (friendly_robots, enemy_robots, balls)
        """
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
        velocities = self._get_object_velocity_at_frame(len(self._records) - 1, object)
        if velocities is None:
            return None
        return self._get_object_velocity_at_frame(len(self._records) - 1, object)

    def _get_object_position_at_frame(self, frame: int, object: GameObject):
        if object == Ball:
            return self._records[frame].ball[0]  # TODO don't always take first ball pos
        elif isinstance(object, RobotEntity):
            if object.colour == Colour.YELLOW:
                return self._records[frame].yellow_robots[object.id]
            else:
                return self._records[frame].blue_robots[object.id]

    def _get_object_velocity_at_frame(
        self, frame: int, object: GameObject
    ) -> Optional[tuple]:
        """
        Calculates the object's velocity based on position changes over time,
          at frame f.

        Returns:
            tuple: The velocity components (vx, vy).

        """
        if frame >= len(self._records) or frame == 0:
            logger.warning("Cannot provide velocity at a frame that does not exist")
            logger.info("See frame: %s", str(frame))
            return None

        # Otherwise get the previous and current frames
        previous_frame = self._records[frame - 1]
        current_frame = self._records[frame]

        previous_pos = self._get_object_position_at_frame(frame - 1, object)
        current_pos = self._get_object_position_at_frame(frame, object)

        previous_time_received = previous_frame.ts
        time_received = current_frame.ts

        # Latest frame should always be ahead of last one
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
                # TODO: Handle when curr_vell is not when (time_received < previous_time_received)
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
        # This function only updates referee records when something changed.

        if not self._referee_records:
            self._referee_records.append(referee_data)

        # TODO: investigate potential namedtuple __eq__  issue
        if referee_data[1:] != self._referee_records[-1][1:]:
            self._referee_records.append(referee_data)

    def source_identifier(self) -> Optional[str]:
        """Get the source identifier."""
        if self._referee_records:
            return self._referee_records[-1].source_identifier
        return None

    @property
    def last_time_sent(self) -> float:
        """Get the time sent."""
        if self._referee_records:
            return self._referee_records[-1].time_sent
        return 0.0

    @property
    def last_time_received(self) -> float:
        """Get the time received."""
        if self._referee_records:
            return self._referee_records[-1].time_received
        return 0.0

    @property
    def last_command(self) -> RefereeCommand:
        """Get the last command."""
        if self._referee_records:
            return self._referee_records[-1].referee_command
        return RefereeCommand.HALT

    @property
    def last_command_timestamp(self) -> float:
        """Get the command timestamp."""
        if self._referee_records:
            return self._referee_records[-1].referee_command_timestamp
        return 0.0

    @property
    def stage(self) -> Stage:
        """Get the current stage."""
        if self._referee_records:
            return self._referee_records[-1].stage
        return Stage.NORMAL_FIRST_HALF_PRE

    @property
    def stage_time_left(self) -> float:
        """Get the time left in the current stage."""
        if self._referee_records:
            return self._referee_records[-1].stage_time_left
        return 0.0

    @property
    def blue_team(self) -> TeamInfo:
        """Get the blue team info."""
        if self._referee_records:
            return self._referee_records[-1].blue_team
        return TeamInfo(
            name="",
            score=0,
            red_cards=0,
            yellow_card_times=[],
            yellow_cards=0,
            timeouts=0,
            timeout_time=0,
            goalkeeper=0,
        )

    @property
    def yellow_team(self) -> TeamInfo:
        """Get the yellow team info."""
        if self._referee_records:
            return self._referee_records[-1].yellow_team
        return TeamInfo(
            name="",
            score=0,
            red_cards=0,
            yellow_card_times=[],
            yellow_cards=0,
            timeouts=0,
            timeout_time=0,
            goalkeeper=0,
        )

    @property
    def designated_position(self) -> Optional[tuple[float]]:
        """Get the designated position."""
        if self._referee_records:
            return self._referee_records[-1].designated_position
        return None

    @property
    def blue_team_on_positive_half(self) -> Optional[bool]:
        """Get the blue team on positive half."""
        if self._referee_records:
            return self._referee_records[-1].blue_team_on_positive_half
        return None

    @property
    def next_command(self) -> Optional[RefereeCommand]:
        """Get the next command."""
        if self._referee_records:
            return self._referee_records[-1].next_command
        return None

    @property
    def current_action_time_remaining(self) -> Optional[int]:
        """Get the current action time remaining."""
        if self._referee_records:
            return self._referee_records[-1].current_action_time_remaining
        return None

    @property
    def is_halt(self) -> bool:
        """Check if the command is HALT."""
        return self.last_command == RefereeCommand.HALT

    @property
    def is_stop(self) -> bool:
        """Check if the command is STOP."""
        return self.last_command == RefereeCommand.STOP

    @property
    def is_normal_start(self) -> bool:
        """Check if the command is NORMAL_START."""
        return self.last_command == RefereeCommand.NORMAL_START

    @property
    def is_force_start(self) -> bool:
        """Check if the command is FORCE_START."""
        return self.last_command == RefereeCommand.FORCE_START

    @property
    def is_prepare_kickoff_yellow(self) -> bool:
        """Check if the command is PREPARE_KICKOFF_YELLOW."""
        return self.last_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    @property
    def is_prepare_kickoff_blue(self) -> bool:
        """Check if the command is PREPARE_KICKOFF_BLUE."""
        return self.last_command == RefereeCommand.PREPARE_KICKOFF_BLUE

    @property
    def is_prepare_penalty_yellow(self) -> bool:
        """Check if the command is PREPARE_PENALTY_YELLOW."""
        return self.last_command == RefereeCommand.PREPARE_PENALTY_YELLOW

    @property
    def is_prepare_penalty_blue(self) -> bool:
        """Check if the command is PREPARE_PENALTY_BLUE."""
        return self.last_command == RefereeCommand.PREPARE_PENALTY_BLUE

    @property
    def is_direct_free_yellow(self) -> bool:
        """Check if the command is DIRECT_FREE_YELLOW."""
        return self.last_command == RefereeCommand.DIRECT_FREE_YELLOW

    @property
    def is_direct_free_blue(self) -> bool:
        """Check if the command is DIRECT_FREE_BLUE."""
        return self.last_command == RefereeCommand.DIRECT_FREE_BLUE

    @property
    def is_timeout_yellow(self) -> bool:
        """Check if the command is TIMEOUT_YELLOW."""
        return self.last_command == RefereeCommand.TIMEOUT_YELLOW

    @property
    def is_timeout_blue(self) -> bool:
        """Check if the command is TIMEOUT_BLUE."""
        return self.last_command == RefereeCommand.TIMEOUT_BLUE

    @property
    def is_ball_placement_yellow(self) -> bool:
        """Check if the command is BALL_PLACEMENT_YELLOW."""
        return self.last_command == RefereeCommand.BALL_PLACEMENT_YELLOW

    @property
    def is_ball_placement_blue(self) -> bool:
        """Check if the command is BALL_PLACEMENT_BLUE."""
        return self.last_command == RefereeCommand.BALL_PLACEMENT_BLUE

    def get_object_velocity(self, object: GameObject):
        return self._get_object_velocity_at_frame(len(self._records) - 1, object)

    def _get_object_position_at_frame(self, frame: int, object: GameObject):
        if object == Ball:
            return self._records[frame].ball[0]  # TODO don't always take first ball pos
        elif isinstance(object, Robot):
            if object.colour == Colour.YELLOW:
                return self._records[frame].yellow_robots[object.id]
            else:
                return self._records[frame].blue_robots[object.id]

    def _get_object_velocity_at_frame(
        self, frame: int, object: GameObject
    ) -> Optional[tuple]:
        """
        Calculates the object's velocity based on position changes over time,
          at frame f.

        Returns:
            tuple: The velocity components (vx, vy).

        """
        if frame >= len(self._records) or frame == 0:
            # Cannot provide velocity at frame that does not exist
            print(frame)
            return None

        # Otherwise get the previous and current frames
        previous_frame = self._records[frame - 1]
        current_frame = self._records[frame]

        previous_pos = self._get_object_position_at_frame(frame - 1, object)
        current_pos = self._get_object_position_at_frame(frame, object)

        previous_time_received = previous_frame.ts
        time_received = current_frame.ts

        # Latest frame should always be ahead of last one
        if time_received < previous_time_received:
            # TODO log a warning
            print("Timestamps out of order for vision data ")
            return None

        dt_secs = time_received - previous_time_received

        vx = (current_pos.x - previous_pos.x) / dt_secs
        vy = (current_pos.y - previous_pos.y) / dt_secs

        return (vx, vy)

    def get_object_acceleration(self, object: GameObject) -> Optional[tuple]:
        totalX = 0
        totalY = 0
        WINDOW = 5
        N_WINDOWS = 6
        iter = 0

        if len(self._records) < WINDOW * N_WINDOWS + 1:
            return None

        for i in range(N_WINDOWS):
            averageVelocity = [0, 0]
            windowStart = 1 + i * WINDOW
            windowEnd = windowStart + WINDOW  # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                curr_vel = self._get_object_velocity_at_frame(
                    len(self._records) - j, object
                )
                averageVelocity[0] += curr_vel[0]
                averageVelocity[1] += curr_vel[1]

            averageVelocity[0] /= WINDOW
            averageVelocity[1] /= WINDOW

            if i != 0:
                dt = (
                    self._records[-windowMiddle + WINDOW].ts
                    - self._records[-windowMiddle].ts
                )
                accX = (futureAverageVelocity[0] - averageVelocity[0]) / dt  # TODO vec
                accY = (futureAverageVelocity[1] - averageVelocity[1]) / dt
                totalX += accX
                totalY += accY
                iter += 1

            futureAverageVelocity = tuple(averageVelocity)

        return (totalX / iter, totalY / iter)

    def predict_object_pos_after(self, t: float, object: GameObject) -> Optional[tuple]:
        # If t is after the object has stopped we return the position at which object stopped.

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

        if ax == 0:  # Due to friction, if acc = 0 then stopped.
            sx = 0  # TODO: Not sure what to do about robots with respect to friction - we never know if they are slowing down to stop or if they are slowing down to change direction
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

        return (
<<<<<<< HEAD
            self.referee_data_handler.last_command == RefereeCommand.BALL_PLACEMENT_BLUE
        )


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
=======
            start_x + sx,
            start_y + sy,
        )  # TODO: Doesn't take into account spin / angular vel

    def predict_frame_after(self, t: float):
        yellow_pos = [
            self.predict_object_pos_after(t, Robot(Colour.YELLOW, i)) for i in range(6)
        ]
        blue_pos = [
            self.predict_object_pos_after(t, Robot(Colour.BLUE, i)) for i in range(6)
        ]
        ball_pos = self.predict_object_pos_after(t, Ball)
        if ball_pos is None or None in yellow_pos or None in blue_pos:
            return None
        else:
            return FrameData(
                self._records[-1].ts + t,
                list(map(lambda pos: RobotData(pos[0], pos[1], 0), yellow_pos)),
                list(map(lambda pos: RobotData(pos[0], pos[1], 0), blue_pos)),
                [BallData(ball_pos[0], ball_pos[1], 0)],  # TODO : Support z axis
            )
>>>>>>> 31d8bcc (Change RefereeCommand and Stage to enum class)
