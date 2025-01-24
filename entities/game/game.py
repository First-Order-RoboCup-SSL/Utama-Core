from typing import List, Optional, NamedTuple

from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData, PredictedFrame
from entities.data.command import RobotInfo

from entities.game.game_object import Colour, GameObject, Robot
from entities.game.game_object import Robot as RobotEntity
from entities.game.robot import Robot
from entities.game.ball import Ball

from team_controller.src.config.settings import TIMESTEP

# TODO : ^ I don't like this circular import logic. Wondering if we should store this constant somewhere else

import logging, warnings

# Configure logging
logger = logging.getLogger(__name__)


class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(
        self,
        my_team_is_yellow=True,
        num_friendly_robots: int = 6,
        num_enemy_robots: int = 6,
    ):
        self._my_team_is_yellow = my_team_is_yellow
        self._field = Field()

        self._records: List[FrameData] = []
        self._predicted_next_frame: PredictedFrame = None

        self._friendly_robots: List[Robot] = [
            Robot(id, is_friendly=True) for id in range(num_friendly_robots)
        ]
        self._enemy_robots: List[Robot] = [
            Robot(id, is_friendly=False) for id in range(num_enemy_robots)
        ]
        self._ball: Ball = Ball()

        self._yellow_score = 0
        self._blue_score = 0

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

    def is_ball_in_goal(self, our_side: bool):
        ball_pos = self.get_ball_pos()[0]
        return (
            ball_pos.x < -self.field.HALF_LENGTH
            and (
                ball_pos.y < self.field.HALF_GOAL_WIDTH
                and ball_pos.y > -self.field.HALF_GOAL_WIDTH
            )
            and not our_side
            or ball_pos.x > self.field.HALF_LENGTH
            and (
                ball_pos.y < self.field.HALF_GOAL_WIDTH
                and ball_pos.y > -self.field.HALF_GOAL_WIDTH
            )
            and our_side
        )

    ### Game state management ###
    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            # if self.my_team_is_yellow:
            #     if len(yellow_robot_data) != len(self._friendly_robots):
            #         logger.warning(f"Expected data for {len(self.friendly_robots)} friendly robots but found {len(yellow_robot_data)} in frame")

            #     if len(blue_robot_data) != len(self._enemy_robots):
            #         logger.warning(f"Expected data for {len(self._enemy_robots)} enemy (blue) robots but found {len(blue_robot_data)} in frame")
            # else:
            #     if len(yellow_robot_data) != len(self._enemy_robots):
            #         logger.warning(f"Expected data for {len(self._enemy_robots)} enemy (yellow) robots but found {len(yellow_robot_data)} in frame")

            #     if len(blue_robot_data) != len(self._friendly_robots):
            #         logger.warning(f"Expected data for {len(self._friendly_robots)} friendly (blue) robots but found {len(blue_robot_data)} in frame")

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
        
        if current_pos is None or previous_pos is None:
            logger.warning("No position data to calculate velocity for frame %d", frame)
            return None

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
                elif missing_velocities == WINDOW - 1:
                    logging.warning(f"No velocity data to calculate acceleration for frame {len(self._records) - j}")
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
