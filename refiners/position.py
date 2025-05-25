from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import vector
import math

from entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from entities.data.vision import VisionBallData, VisionData, VisionRobotData
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.robot import Robot
from refiners.base_refiner import BaseRefiner
from global_utils.math_utils import normalise_heading
from collections import defaultdict


class AngleSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor for angle
        self.smoothed_angles = {}  # Stores last smoothed angle for each robot

    def smooth(self, old_angle: float, new_angle: float) -> float:
        # Compute the shortest angular difference
        diff = math.atan2(
            math.sin(new_angle - old_angle), math.cos(new_angle - old_angle)
        )
        smoothed_angle = old_angle + self.alpha * diff

        return normalise_heading(smoothed_angle)


class PositionRefiner(BaseRefiner):
    def __init__(self):
        self.angle_smoother = AngleSmoother(alpha=0.4)
        self.half_field_length = 4.5  # Example field length, adjust as needed
        self.half_field_width = 3.0  # Example field width, adjust as needed

    # Primary function for the Refiner interface
    def refine(self, game: Game, data: List[RawVisionData]):

        self.half_field_length = game.field.half_length
        self.half_field_width = game.field.half_width

        data = [frame for frame in data if frame is not None]

        # If no information just return the original
        if not data:
            return game
        # Can combine previous position from game with new data to produce new position if desired
        combined_vision_data = CameraCombiner().combine_cameras(game, data)

        new_yellow_robots, new_blue_robots = (
            self._combine_both_teams_game_vision_positions(
                game,
                combined_vision_data.yellow_robots,
                combined_vision_data.blue_robots,
            )
        )

        # After the balls have been combined, take the most confident
        new_ball = PositionRefiner._get_most_confident_ball(combined_vision_data.balls)
        if new_ball is None:
            # If none, take the ball from the last frame of the game
            new_ball = game.ball

        if game.my_team_is_yellow:
            new_game = replace(
                game,
                friendly_robots=new_yellow_robots,
                enemy_robots=new_blue_robots,
                ball=new_ball,
            )
        else:
            new_game = replace(
                game,
                friendly_robots=new_blue_robots,
                enemy_robots=new_yellow_robots,
                ball=new_ball,
            )

        return new_game

    # Static methods
    @staticmethod
    def _combine_robot_vision_data(
        old_robot: Robot, robot_data: VisionRobotData, angle_smoother: AngleSmoother
    ) -> Robot:
        assert old_robot.id == robot_data.id
        new_x, new_y = robot_data.x, robot_data.y

        if (
            new_x > PositionRefiner.half_field_length
            or new_x < -PositionRefiner.half_field_length
        ):
            new_x = old_robot.p.x  # Keep old x if new x is invalid
        if (
            new_y > PositionRefiner.half_field_width
            or new_y < -PositionRefiner.half_field_width
        ):
            new_y = old_robot.p.y  # Keep old y if new y is invalid

        # Smoothing
        new_orientation = angle_smoother.smooth(
            old_robot.orientation, robot_data.orientation
        )
        return replace(
            old_robot,
            id=robot_data.id,
            p=vector.obj(x=new_x, y=new_y),
            orientation=new_orientation,
        )

    # Used at start of the game so assume robot does not have the ball
    # Also assume velocity and acceleration are zero
    @staticmethod
    def _robot_from_vision(robot_data: VisionRobotData, is_friendly: bool) -> Robot:
        return Robot(
            id=robot_data.id,
            is_friendly=is_friendly,
            has_ball=False,
            p=vector.obj(x=robot_data.x, y=robot_data.y),
            v=vector.obj(x=0, y=0),
            a=vector.obj(x=0, y=0),
            orientation=robot_data.orientation,
        )

    @staticmethod
    def _ball_from_vision(ball_data: VisionBallData) -> Ball:
        zv = vector.obj(x=0, y=0, z=0)
        return Ball(vector.obj(x=ball_data.x, y=ball_data.y, z=ball_data.z), zv, zv)

    @staticmethod
    def _get_most_confident_ball(balls: List[VisionBallData]) -> Ball:
        balls_by_confidence = sorted(
            balls, key=lambda ball: ball.confidence, reverse=True
        )
        if not balls_by_confidence:
            return None
        return PositionRefiner._ball_from_vision(balls_by_confidence[0])

    def _combine_single_team_positions(
        self,
        game_robots: Dict[int, Robot],
        vision_robots: List[VisionRobotData],
        friendly: bool,
    ) -> Dict[int, Robot]:
        new_game_robots = game_robots.copy()
        for robot in vision_robots:
            if robot.id not in new_game_robots:
                # At the start of the game, we haven't seen anything yet, so just create a new robot
                new_game_robots[robot.id] = PositionRefiner._robot_from_vision(
                    robot, is_friendly=friendly
                )
            else:
                # Update with smoothed data.
                new_game_robots[robot.id] = PositionRefiner._combine_robot_vision_data(
                    new_game_robots[robot.id], robot, self.angle_smoother
                )
        return new_game_robots

    def _combine_both_teams_game_vision_positions(
        self,
        game: Game,
        yellow_vision_robots: List[VisionRobotData],
        blue_vision_robots: List[VisionRobotData],
    ) -> Tuple[Dict[int, Robot], Dict[int, Robot]]:

        if game.my_team_is_yellow:
            old_yellow_robots = game.friendly_robots.copy()
            old_blue_robots = game.enemy_robots.copy()
        else:
            old_yellow_robots = game.enemy_robots.copy()
            old_blue_robots = game.friendly_robots.copy()

        new_yellow_robots = self._combine_single_team_positions(
            old_yellow_robots, yellow_vision_robots, friendly=game.my_team_is_yellow
        )
        new_blue_robots = self._combine_single_team_positions(
            old_blue_robots, blue_vision_robots, friendly=not game.my_team_is_yellow
        )

        return new_yellow_robots, new_blue_robots


class CameraCombiner:

    BALL_CONFIDENCE_THRESHOLD = 0.1
    BALL_MERGE_THRESHOLD = 0.05

    def combine_cameras(self, game: Game, frames: List[RawVisionData]) -> VisionData:
        # Now we have access to the game we can do more sophisticated things
        # Such as ignoring outlier cameras etc

        ts = []
        # maps robot id to list of frames seen for that robot
        yellow_captured = defaultdict(list)
        blue_captured = defaultdict(list)
        balls_captured = defaultdict(list)

        # Each frame is from a different camera
        for frame_ind, frame in enumerate(frames):
            for yr in frame.yellow_robots:
                yellow_captured[yr.id].append(yr)

            for br in frame.blue_robots:
                blue_captured[br.id].append(br)

            for b in frame.balls:
                balls_captured[frame_ind].append(b)
            ts.append(frame.ts)

        avg_yellows = list(map(self._avg_robots, yellow_captured.values()))
        avg_blues = list(map(self._avg_robots, blue_captured.values()))
        # Current strategy is just to take the most confident ball
        balls = self._combine_balls_by_proximity(balls_captured)

        return VisionData(sum(ts) / len(ts), avg_yellows, avg_blues, balls)

    def _avg_robots(self, rs: List[RawRobotData]) -> Optional[VisionRobotData]:
        # All these robots should have the same id
        if not rs:
            return None
        base_id = rs[0].id
        tx, ty, to, tc = 0, 0, 0, 0
        for r in rs:
            assert base_id == r.id
            tx += r.x
            ty += r.y
            to += r.orientation
            tc += r.confidence

        return VisionRobotData(base_id, tx / len(rs), ty / len(rs), to / len(rs))

    def _combine_balls_by_proximity(
        self, bs: Dict[int, List[RawBallData]]
    ) -> List[VisionBallData]:
        combined_balls: List[VisionBallData] = []
        for ball_list in bs.values():
            for b in ball_list:
                if b.confidence > CameraCombiner.BALL_CONFIDENCE_THRESHOLD:
                    found = False
                    for i, cb in enumerate(combined_balls):
                        if CameraCombiner.ball_merge_predicate(b, cb):
                            found = True
                            combined_balls[i] = CameraCombiner.ball_merge(cb, b)
                            break

                    if not found:
                        # If no ball close enough, must have found a new separate ball
                        combined_balls.append(b)
        return combined_balls

    def ball_merge_predicate(b1: RawBallData, b2: RawBallData) -> bool:
        return abs(b1.x - b2.x) + abs(b1.y - b2.y) < CameraCombiner.BALL_MERGE_THRESHOLD

    def ball_merge(b1: RawBallData, b2: RawBallData) -> RawBallData:
        nx = (b1.x + b2.x) / 2
        ny = (b1.y + b2.y) / 2
        nz = (b1.z + b2.z) / 2
        nc = max(b1.confidence, b2.confidence)
        return RawBallData(nx, ny, nz, nc)
