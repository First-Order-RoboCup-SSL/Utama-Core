

from dataclasses import replace
from typing import List, Optional
from entities.data.raw_vision import RawVisionData
from entities.data.vision import VisionBallData, VisionData, VisionRobotData
from entities.game.game import Game
from entities.game.robot import Robot
from refiners.base_refiner import BaseRefiner

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

def ball_from_vision(ball_data: VisionBallData) -> Ball:
    return Ball(ball_data.x, ball_data.y, ball_data.z)



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


