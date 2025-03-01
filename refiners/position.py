

from typing import List, Optional
from entities.data.raw_vision import RawVisionData
from entities.data.vision import VisionBallData, VisionData, VisionRobotData
from entities.game.game import Game
from refiners.base_refiner import BaseRefiner


class PositionRefiner(BaseRefiner):

    def refine(self, game: Game, data: List[RawVisionData]):
        # Can combine previous position from game with new data to produce new position if desired

        processed_vision_data = CameraCombiner().combine_cameras(game, data)
        # old_game_positions = game.get_old_positions() 
        game.set_new_positions()
        return game
    

class CameraCombiner:

    def combine_cameras(self, game: Game, data: List[RawVisionData]) -> VisionData:
        # Now we have access to the game we can do more sophisticated things
        # Such as ignoring outlier cameras etc
        return self._avg_frames(data)
    
    def _avg_robots(self, rs: List[VisionRobotData]) -> Optional[VisionRobotData]:
        if not rs:
            return None

        tx, ty, to = 0, 0, 0
        for r in rs:
            tx += r.x
            ty += r.y
            to += r.orientation

        return VisionRobotData(tx / len(rs), ty / len(rs), to / len(rs))

    def _avg_balls(self, bs: List[VisionBallData]) -> Optional[VisionBallData]:
        if not bs:
            return None

        tx, ty, tz = 0, 0, 0
        for r in bs:
            tx += r.x
            ty += r.y
            tz += r.z

        return VisionBallData(
            tx / len(bs),
            ty / len(bs),
            tz / len(bs),
            min(map(lambda x: x.confidence, bs)),
        )

    def _avg_frames(self, frames: List[RawVisionData]) -> VisionData:
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

        return VisionData(ts, avg_yellows[:-5], avg_blues[:-5], avg_balls[:-10])


