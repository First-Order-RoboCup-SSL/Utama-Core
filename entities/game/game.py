from typing import List, Optional
from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData

class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(self):
        self._field = Field()
        self._records = []
        self._yellow_score = 0
        self._blue_score = 0

    @property
    def field(self) -> Field:
        return self._field

    @property
    def current_state(self) -> FrameData:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> list[FrameData]:
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

    def get_ball_pos(self) -> BallData:
        return self._records[-1].ball

    def get_ball_velocity(self):
        return self._get_ball_velocity_at_frame(len(self._records) - 1)

    def _get_ball_velocity_at_frame(self, frame: int) -> Optional[tuple]:
        """
        Calculates the ball's velocity based on position changes over time,
          at frame f.

        Returns:
            tuple: The velocity components (vx, vy).

        """
        if frame >= len(self._records) or frame == 0:
            # Cannot provide velocity at frame that does not exist
            return None
        
        # Otherwise get the previous and current frames
        previous_frame = self._records[frame - 1]
        current_frame = self._records[frame]
        
        previous_ball_pos = previous_frame.ball[0] # TODO don't always take first ball pos
        ball_pos = current_frame.ball[0] # TODO don't always take first ball pos

        previous_time_received = previous_frame.ts
        time_received = current_frame.ts

        # Latest frame should always be ahead of last one    
        if time_received < previous_time_received:
            # TODO log a warning
            print("Timestamps out of order for vision data ")
            return None        
        
        dt_secs = time_received - previous_time_received
        
        vx = (ball_pos.x - previous_ball_pos.x) / dt_secs
        vy = (ball_pos.y - previous_ball_pos.y) / dt_secs
        
        return (vx, vy) ## mm/sec

    def get_ball_acceleration(self) -> Optional[tuple]:
        if len(self._records) < 10:
            return None
        
        totalX = 0
        totalY = 0
        for i in range(1, 6):
            curr_vel = self._get_ball_velocity_at_frame(len(self._records) - i)
            prev_vel = self._get_ball_velocity_at_frame(len(self._records) - i - 1)
            
            dt = self._records[-i].ts - self._records[-i - 1].ts

            accX = (curr_vel[0] - prev_vel[0]) / dt    # TODO vec
            accY = (curr_vel[1] - prev_vel[1]) / dt
            # print(accX, accY)
            # print(prev_vel, curr_vel, dt)
            totalX += accX
            totalY += accY
        return (totalX / 5, totalY / 5) ## mm/(sec^2)

    def predict_ball_pos_after(self, t: float) -> Optional[tuple]: # t in secs
        # If t is after the ball has stopped we return the position at which ball stopped.
        
        acc = self.get_ball_acceleration()
        
        if acc is None:
            return None
        
        ax, ay = acc
        ux, uy = self.get_ball_velocity()
        ball = self.get_ball_pos()
        start_x, start_y = ball[0].x, ball[0].y

        # print(acc)
        # print(self.get_ball_velocity())

        if ax == 0: # Due to friction, if acc = 0 then stopped. 
            sx = 0
        else:
            tx_stop = - ux / ax
            tx = min(t, tx_stop)
            sx = ux * tx + 0.5 * ax * tx * tx # mm

        if ay == 0:
            sy = 0
        else:
            ty_stop = - uy / ay
            ty = min(t, ty_stop)
            sy = uy * ty + 0.5 * ay * ty * ty # mm

        return (start_x + sx, start_y + sy) # TODO: Doesn't take into account spin / angular vel
    