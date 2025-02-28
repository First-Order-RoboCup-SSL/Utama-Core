import time
from typing import Dict, List, Optional
from config.settings import TIMESTEP
from entities.data.raw_vision import RawFrameData
from entities.data.vision import BallData, FrameData, RobotData
from queue import SimpleQueue

from team_controller.src.data.message_enum import MessageType

class VisionProcessor:
    """
    Puts processed vision data into queue:
        - 60 fps
        - No empty values (all extrapolated)
        - Fixes out of order packets
        - Averages multiple cameras
    Only starts when we have enough data
    """
    def __init__(self,
                 n_expected_robots_yellow: int,
                 n_expected_robots_blue: int,
                 n_expected_balls: int,
                 queue: SimpleQueue):
        self.n_expected_robots_yellow = n_expected_robots_yellow
        self.n_expected_robots_blue = n_expected_robots_blue
        self.n_expected_balls = n_expected_balls
        self.queue = queue
        self.last_enqueue_time: float = 0     # timestamp of last enqueue operation
        self.highest_seen_packet = None       # highest timestamp packet
        self.camera_views: Dict[int, RawFrameData] = {} 

    def is_ready(self) -> bool:
        frame = self._combine_cameras()
        return (len(frame.blue_robots) == self.n_expected_robots_blue
               and len(frame.yellow_robots) == self.n_expected_robots_yellow 
               and len(frame.n_expected_balls) == self.n_expected_balls)

    def _combine_cameras(self) -> FrameData:
        return self._avg_frames([*self.camera_views.values()])
    
    def _extrapolate(self, frame: FrameData) -> FrameData:
        # TODO : Extrapolation
        return frame

    def add_new_frame(self, frame_data: RawFrameData):
        if frame_data.camera_id not in self.camera_views or frame_data.ts >= self.camera_views[frame_data.camera_id].ts:
            self.camera_views[frame_data.camera_id] = frame_data
        
        should_create_new_processed_frame = time.time() - self.last_enqueue_time > TIMESTEP and self.is_ready()

        if should_create_new_processed_frame:
            self.queue.push((MessageType.VISION, self._extrapolate(self._combine_cameras())))
            self.last_enqueue_time = time.time() 

        if frame_data.ts > self.highest_seen_packet.ts:
            self.highest_seen_packet = frame_data

    def _avg_robots(self, rs: List[RobotData]) -> Optional[RobotData]:
        if not rs:
            return None

        tx, ty, to = 0, 0, 0
        for r in rs:
            tx += r.x
            ty += r.y
            to += r.orientation

        return RobotData(tx / len(rs), ty / len(rs), to / len(rs))

    def _avg_balls(self, bs: List[BallData]) -> Optional[BallData]:
        if not bs:
            return None

        tx, ty, tz = 0, 0, 0
        for r in bs:
            tx += r.x
            ty += r.y
            tz += r.z

        return BallData(
            tx / len(bs),
            ty / len(bs),
            tz / len(bs),
            min(map(lambda x: x.confidence, bs)),
        )

    def _avg_frames(self, frames: List[FrameData]) -> FrameData:
        frames = [*filter(lambda x: x.ball is not None, frames)]
        ts = 0
        yellow_captured = [[] for _ in range(11)]
        blue_captured = [[] for _ in range(11)]
        ball_captured = [[] for _ in range(11)]

        for frame in frames:
            for ind, yr in enumerate(frame.yellow_robots):
                if yr is not None:
                    yellow_captured[ind].append(yr)
            for ind, br in enumerate(frame.blue_robots):
                if br is not None:
                    blue_captured[ind].append(br)
            for ind, b in enumerate(frame.ball):
                if b is not None:
                    ball_captured[ind].append(b)
            ts += frame.ts

        avg_yellows = list(map(self._avg_robots, yellow_captured))
        avg_blues = list(map(self._avg_robots, blue_captured))
        avg_balls = list(map(self._avg_balls, ball_captured))

        # Trims number of robots in frame to number we expect (num_friendly, num_enemy) currently done to 6
        return FrameData(ts, avg_yellows[:-5], avg_blues[:-5], avg_balls[:-10])
