from dataclasses import replace
from typing import Dict, List, Optional, Tuple
from entities.data.raw_vision import RawVisionData
from entities.data.vision import VisionBallData, VisionData, VisionRobotData
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.robot import Robot
from refiners.base_refiner import BaseRefiner

class PositionRefiner(BaseRefiner):
    def _combine_robot_vision_data(old_robot: Robot, robot_data: VisionRobotData) -> Robot:
        assert old_robot.id == robot_data.id
        return replace(old_robot,
            id=robot_data.id,
            x=robot_data.x,
            y=robot_data.y,
            orientation=robot_data.orientation,
        )

    # Used at start of the game so assume robot does not have the ball
    def _robot_from_vision(robot_data: VisionRobotData, is_friendly: bool) -> Robot:
        return Robot(
            id=robot_data.id,
            is_friendly=is_friendly,
            has_ball=False,
            x=robot_data.x,
            y=robot_data.y,
            orientation=robot_data.orientation,
        )

    def _ball_from_vision(ball_data: VisionBallData) -> Ball:
        return Ball(ball_data.x, ball_data.y, ball_data.z)

    def _get_most_confident_ball(balls: List[VisionBallData]) -> Ball:
        balls_by_confidence = sorted(
            balls, key=lambda ball: ball.confidence, reverse=True
        )
        return Ball(
            balls_by_confidence[0].x, balls_by_confidence[0].y, balls_by_confidence[0].z
        )

    def _combine_single_team_positions(self, game_robots:Dict[int, Robot], vision_robots: List[VisionRobotData], friendly: bool) -> Dict[int, Robot]:
        new_game_robots = dict(game_robots)
        for robot in vision_robots:
            if robot.id not in new_game_robots:
                # At the start of the game, we haven't seen anything yet, so just create a new robot
                new_game_robots[robot.id] = PositionRefiner._robot_from_vision(robot, is_friendly=friendly)
            else:
                # Otherwise we have old information so we update it
                new_game_robots[robot.id] = PositionRefiner._combine_robot_vision_data(new_game_robots[robot.id], robot)

        return new_game_robots

    def _combine_both_teams_game_vision_positions(self, game: Game, yellow_vision_robots: List[VisionRobotData], blue_vision_robots: List[VisionRobotData]) -> Tuple[Dict[int, Robot], Dict[int, Robot]]:
        
        if game.my_team_is_yellow:
            old_yellow_robots = dict(game.friendly_robots)
            old_blue_robots = dict(game.enemy_robots)
        else:
            old_yellow_robots = dict(game.enemy_robots)
            old_blue_robots = dict(game.friendly_robots)
        
        new_yellow_robots = self._combine_single_team_positions(old_yellow_robots, yellow_vision_robots, friendly=game.my_team_is_yellow)
        new_blue_robots = self._combine_single_team_positions(old_blue_robots, blue_vision_robots, friendly=not game.my_team_is_yellow)

        return new_yellow_robots, new_blue_robots

    def refine(self, game: Game, data: List[RawVisionData]):
        # Can combine previous position from game with new data to produce new position if desired
        combined_vision_data = CameraCombiner().combine_cameras(game, data)

        new_yellow_robots, new_blue_robots = self._combine_both_teams_game_vision_positions(game, combined_vision_data.yellow_robots, combined_vision_data.blue_robots)
        
        # Same thing here with ball, instead of using most confident, we can look in game to see 
        # which new vision ball is closest to the game ball and take that 
        new_ball = PositionRefiner._ball_from_vision(self._get_most_confident_ball(combined_vision_data.balls))

        if game.my_team_is_yellow:
            new_game = replace(game, friendly_robots=new_yellow_robots, enemy_robots=new_blue_robots, ball=new_ball)
        else:
            new_game = replace(game, friendly_robots=new_blue_robots, enemy_robots=new_yellow_robots, ball=new_ball)

        return new_game
    

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


