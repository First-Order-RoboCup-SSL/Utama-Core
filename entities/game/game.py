from typing import List, Optional
from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData
from enum import Enum

from entities.game.game_object import Ball, Colour, GameObject, Robot


class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(self):
        self._field = Field()
        self._records = []
        self._yellow_score = 0
        self._blue_score = 0

    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            self._records.append(frame_data)
        else:
            raise ValueError("Invalid frame data.")

    def get_robots_pos(self, is_yellow: bool) -> List[RobotData]:
        if not self._records:
            return None
        record = self._records[-1]
        return record.yellow_robots if is_yellow else record.blue_robots

    def get_ball_pos(self) -> BallData:
        if not self._records:
            return None
        return self._records[-1].ball

    def get_latest_frame(self) -> FrameData:
        if not self._records:
            return None
        return self._records[-1]

    @property
    def field(self) -> Field:
        return self._field

    @property
    def current_state(self) -> FrameData:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> list[FrameData]:
        if not self._records:
            return None
        return self._records

    @property
    def yellow_score(self) -> int:
        return self._yellow_score

    @property
    def blue_score(self) -> int:
        return self._blue_score

    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            self._records.append(frame_data)
        else:
            raise ValueError("Invalid frame data.")
    
    def get_robots_pos(self, is_yellow: bool) -> List[RobotData]:
        record = self._records[-1] 
        return record.yellow_robots if is_yellow else record.blue_robots

    def get_object_velocity(self, object: GameObject):
        return self._get_object_velocity_at_frame(len(self._records) - 1, object)

    def _get_object_position_at_frame(self, frame: int, object: GameObject):
        if object == Ball:
            return self._records[frame].ball[0] # TODO don't always take first ball pos
        elif isinstance(object, Robot):
            if object.colour == Colour.YELLOW:
                return self._records[frame].yellow_robots[object.id]
            else:
                return self._records[frame].blue_robots[object.id]

    def _get_object_velocity_at_frame(self, frame: int, object: GameObject) -> Optional[tuple]:
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
            windowEnd = windowStart + WINDOW # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                curr_vel = self._get_object_velocity_at_frame(len(self._records) - j, object)
                averageVelocity[0] += curr_vel[0]
                averageVelocity[1] += curr_vel[1]
            
            averageVelocity[0] /= WINDOW
            averageVelocity[1] /= WINDOW

            if i != 0:
                dt = self._records[-windowMiddle + WINDOW].ts - self._records[- windowMiddle].ts
                accX = (futureAverageVelocity[0] - averageVelocity[0]) / dt    # TODO vec
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

        if ax == 0: # Due to friction, if acc = 0 then stopped. 
            sx = 0 # TODO: Not sure what to do about robots with respect to friction - we never know if they are slowing down to stop or if they are slowing down to change direction 
        else:
            tx_stop = - ux / ax
            tx = min(t, tx_stop)
            sx = ux * tx + 0.5 * ax * tx * tx

        if ay == 0:
            sy = 0
        else:
            ty_stop = - uy / ay
            ty = min(t, ty_stop)
            sy = uy * ty + 0.5 * ay * ty * ty

        return (start_x + sx, start_y + sy) # TODO: Doesn't take into account spin / angular vel
    
    def predict_frame_after(self, t: float):
        yellow_pos = [self.predict_object_pos_after(t, Robot(Colour.YELLOW, i)) for i in range(6)]
        blue_pos = [self.predict_object_pos_after(t, Robot(Colour.BLUE, i)) for i in range(6)]
        ball_pos = self.predict_object_pos_after(t, Ball)
        if ball_pos is None or None in yellow_pos or None in blue_pos:
            return None
        else:
            return FrameData(
                self._records[-1].ts + t,
                list(map(lambda pos: RobotData(pos[0], pos[1], 0), yellow_pos)),            
                list(map(lambda pos: RobotData(pos[0], pos[1], 0), blue_pos)),
                [BallData(ball_pos[0], ball_pos[1], 0)] # TODO : Support z axis            
            )