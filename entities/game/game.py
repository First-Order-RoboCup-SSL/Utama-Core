from typing import List, Optional

from numpy import empty
from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData
from entities.data.command import RobotInfo

from entities.game.game_object import Ball, Colour, GameObject, Robot
from entities.game.robot import Friendly, Enemy

from team_controller.src.config.settings import TIMESTEP

# TODO : ^ I don't like this circular import logic. Wondering if we should store this constant somewhere else

class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(self, my_team_is_yellow=True):
        self._field = Field(my_team_is_yellow)
        self._records: List[RobotData] = []
        self._predicted_next_frame = None
        self._my_team_is_yellow = my_team_is_yellow
        self._friendly_robots: List[Friendly] = [Friendly(id) for id in range(6)]
        self._enemy_robots: List[Enemy] = [Enemy(id) for id in range(6)]
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
    def predicted_next_frame(self) -> FrameData:
        return self._predicted_next_frame
    
    @property
    def friendly_robots(self) -> List[Friendly]:
        return self._friendly_robots
    
    @friendly_robots.setter
    def friendly_robots(self, value: List[RobotData]):
        for robot_id, robot_data in enumerate(value):
            self._friendly_robots[robot_id].robot_data = robot_data    
        
    @property
    def enemy_robots(self) -> List[Enemy]:
        return self._enemy_robots
    
    @enemy_robots.setter
    def enemy_robots(self, value: List[RobotData]):
        for robot_id, robot_data in enumerate(value):
            self._enemy_robots[robot_id].robot_data = robot_data

    ### Game state management ###
    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            self._records.append(frame_data)
            self._predicted_next_frame = self.predict_frame_after(TIMESTEP)
            self._update_robots(frame_data)
        else:
            raise ValueError("Invalid frame data.")
    
    def add_robot_info(self, robots_info: List[RobotInfo]) -> None:
        for robot_id, robot_info in enumerate(robots_info):
            self._friendly_robots[robot_id].has_ball = robot_info.has_ball
            # Extensible with more info (remeber to add the property in robot.py)
            
    def _update_robots(self, frame_data: FrameData) -> None:
        if self.my_team_is_yellow:
            self.friendly_robots = frame_data.yellow_robots
            self.enemy_robots = frame_data.blue_robots
        else:
            self.friendly_robots = frame_data.blue_robots
            self.enemy_robots = frame_data.yellow_robots

    ### Robot data retrieval ###
    def get_robots_pos(self, is_yellow: bool) -> List[RobotData]:
        if not self._records:
            return None
        record = self._records[-1]
        return record.yellow_robots if is_yellow else record.blue_robots

    def get_robots_velocity(self, is_yellow: bool) -> List[tuple]:
        """
        Returns (vx, vy) of all robots on a team at the latest frame. None if no data available.
        """
        if len(self._records) <= 1:
            return None
        if is_yellow:
            # TODO: potential namespace conflict when robot (robot.py) entity is reintroduced. Think about integrating the two
            return [
                self.get_object_velocity(Robot(i, Colour.YELLOW))
                for i in range(
                    len(self.get_robots_pos(True))
                )  # TODO: This is a bit of a hack, we should be able to get the number of robots from the field
            ]
        else:
            return [
                self.get_object_velocity(Robot(i, Colour.BLUE))
                for i in range(len(self.get_robots_pos(False)))
            ]

    ### Ball Data retrieval ###
    def get_ball_pos(self) -> BallData:
        if not self._records:
            return None
        return self._records[-1].ball

    def get_ball_velocity(self) -> Optional[tuple]:
        """
        Returns (vx, vy) of the ball at the latest frame. None if no data available.
        """
        return self.get_object_velocity(Ball)

    ### Frame Data retrieval ###
    def get_latest_frame(self) -> tuple[RobotData, RobotData, BallData]:
        """
        FrameData rearranged as (friendly_robots, enemy_robots, balls) based on provided _my_team_is_yellow field
        """
        if not self._records:
            return None
        latest_frame = self._records[-1]
        return self._reorganise_frame_data(latest_frame)
    
    def predict_frame_after(self, t: float) -> FrameData:
        """
        Predicts frame in t seconds from the latest frame.
        """
        yellow_pos = [
            self.predict_object_pos_after(t, Robot(Colour.YELLOW, i))
            for i in range(len(self.get_robots_pos(True)))
        ]
        blue_pos = [
            self.predict_object_pos_after(t, Robot(Colour.BLUE, i))
            for i in range(len(self.get_robots_pos(False)))
        ]
        ball_pos = self.predict_object_pos_after(t, Ball)
        if ball_pos is None or None in yellow_pos or None in blue_pos:
            return None
        else:
            return self._reorganise_frame_data(
                FrameData(
                    self._records[-1].ts + t,
                    list(map(lambda pos: RobotData(pos[0], pos[1], 0), yellow_pos)),
                    list(map(lambda pos: RobotData(pos[0], pos[1], 0), blue_pos)),
                    [BallData(ball_pos[0], ball_pos[1], 0)],  # TODO : Support z axis
                )
            )

    def _reorganise_frame_data(
        self, frame_data: FrameData
    ) -> tuple[RobotData, RobotData, BallData]:
        """
        reorganises frame data to be (friendly_robots, enemy_robots, balls)
        """
        _, yellow_robots, blue_robots, balls = frame_data
        if self._my_team_is_yellow:
            return yellow_robots, blue_robots, balls
        else:
            return blue_robots, yellow_robots, balls

    ### General Object Position Prediction ###

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
            start_x + sx,
            start_y + sy,
        )  # TODO: Doesn't take into account spin / angular vel

    def get_object_velocity(self, object: GameObject) -> Optional[tuple]:
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
