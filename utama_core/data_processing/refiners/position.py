from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from utama_core.config.settings import BALL_MERGE_THRESHOLD
from utama_core.data_processing.refiners.base_refiner import BaseRefiner
from utama_core.data_processing.refiners.filters.kalman import (
    KalmanFilter,
    KalmanFilterBall,
)
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.data.vision import VisionBallData, VisionData, VisionRobotData
from utama_core.entities.game import Ball, FieldBounds, GameFrame, Robot
from utama_core.global_utils.mapping_utils import map_friendly_enemy_to_colors


class AngleSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor for angle
        self.smoothed_angles = {}  # Stores last smoothed angle for each robot

    def smooth(self, old_angle: float, new_angle: float) -> float:
        # Compute the shortest angular difference
        diff = np.atan2(np.sin(new_angle - old_angle), np.cos(new_angle - old_angle))
        smoothed_angle = old_angle + self.alpha * diff

        return smoothed_angle


@dataclass
class VisionBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class PositionRefiner(BaseRefiner):
    def __init__(
        self,
        field_bounds: FieldBounds,
        bounds_buffer: float = 1.0,
        filtering: bool = True,
    ):
        # alpha=0 means no change in angle (inf smoothing), alpha=1 means no smoothing
        self.angle_smoother = AngleSmoother(alpha=1)
        self.vision_bounds = VisionBounds(
            x_min=field_bounds.top_left[0] - bounds_buffer,  # expand left
            x_max=field_bounds.bottom_right[0] + bounds_buffer,  # expand right
            y_min=field_bounds.bottom_right[1] - bounds_buffer,  # expand bottom
            y_max=field_bounds.top_left[1] + bounds_buffer,  # expand top
        )

        # For Kalman filtering and imputing vanished values.
        self.filtering = filtering
        self._filter_running = (
            False  # Only start filtering once we have valid data to filter (i.e. after the first valid game frame)
        )

        if self.filtering:
            # Instantiate a dedicated Kalman filter for each robot so filtering can be kept independent.
            self.kalman_filters_yellow: dict[int, KalmanFilter] = {}
            self.kalman_filters_blue: dict[int, KalmanFilter] = {}
            self.kalman_filter_ball = KalmanFilterBall()

    # Primary function for the Refiner interface
    def refine(self, game_frame: GameFrame, data: List[RawVisionData]) -> GameFrame:
        frames = [frame for frame in data if frame is not None]

        # If no information just return the original
        if not frames:
            return game_frame

        # class VisionData: ts: float; yellow_robots: List[VisionRobotData]; blue_robots: List[VisionRobotData]; balls: List[VisionBallData]
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        combined_vision_data: VisionData = CameraCombiner().combine_cameras(frames, bounds=self.vision_bounds)

        time_elapsed = combined_vision_data.ts - game_frame.ts

        # For filtering and vanishing
        if self.filtering and self._filter_running:  # Checks if the first valid game frame has been received.
            # For vanishing: imputes combined_vision_data with null vision frames in place.
            vision_yellow, vision_blue = self._include_vanished_robots(combined_vision_data, game_frame)

            yellow_rbt_last_frame, blue_rbt_last_frame = map_friendly_enemy_to_colors(
                game_frame.my_team_is_yellow,
                game_frame.friendly_robots,
                game_frame.enemy_robots,
            )

            filtered_yellow_robots = []
            for y_rbt_id, vision_y_rbt in vision_yellow.items():
                if y_rbt_id not in self.kalman_filters_yellow:
                    self.kalman_filters_yellow[y_rbt_id] = KalmanFilter(id=y_rbt_id)
                    if y_rbt_id not in yellow_rbt_last_frame:
                        filtered_yellow_robots.append(vision_y_rbt)
                        continue
                filtered_robot = self.kalman_filters_yellow[y_rbt_id].filter_data(
                    vision_y_rbt,  # new measurement
                    yellow_rbt_last_frame[y_rbt_id],  # last frame
                    time_elapsed,
                )

                filtered_yellow_robots.append(filtered_robot)

            filtered_blue_robots = []
            for b_rbt_id, vision_b_rbt in vision_blue.items():
                if b_rbt_id not in self.kalman_filters_blue:
                    self.kalman_filters_blue[b_rbt_id] = KalmanFilter(id=b_rbt_id)
                    if b_rbt_id not in blue_rbt_last_frame:
                        filtered_blue_robots.append(vision_b_rbt)
                        continue

                filtered_robot = self.kalman_filters_blue[b_rbt_id].filter_data(
                    vision_b_rbt,  # new measurement
                    blue_rbt_last_frame[b_rbt_id],  # last frame
                    time_elapsed,
                )

                filtered_blue_robots.append(filtered_robot)

            combined_vision_data = VisionData(
                ts=combined_vision_data.ts,
                yellow_robots=filtered_yellow_robots,
                blue_robots=filtered_blue_robots,
                balls=combined_vision_data.balls,
            )

        # Some processing of robot vision data
        new_yellow_robots, new_blue_robots = self._combine_both_teams_game_vision_positions(
            game_frame,
            combined_vision_data.yellow_robots,
            combined_vision_data.blue_robots,
        )

        # After the balls have been combined, take the most confident
        new_ball: Ball = PositionRefiner._get_most_confident_ball(combined_vision_data.balls)

        # For filtering and vanishing
        if self.filtering and self._filter_running:
            new_ball = self.kalman_filter_ball.filter_data(
                new_ball,
                game_frame.ball,
                time_elapsed,
            )
        elif new_ball is None:
            # If none, take the ball from the last frame of the game
            new_ball = game_frame.ball

        if game_frame.my_team_is_yellow:
            new_game_frame = replace(
                game_frame,
                ts=combined_vision_data.ts,
                friendly_robots=new_yellow_robots,
                enemy_robots=new_blue_robots,
                ball=new_ball,
            )
        else:
            new_game_frame = replace(
                game_frame,
                ts=combined_vision_data.ts,
                friendly_robots=new_blue_robots,
                enemy_robots=new_yellow_robots,
                ball=new_ball,
            )

        return new_game_frame

    def reset(self):
        """
        Resets the internal state of the refiner, including Kalman filters and vanishing trackers.
        Should be called at the start of each game to ensure no leakage of information between games.
        """
        self._filter_running = False
        if self.filtering:
            self.kalman_filters_yellow = {}
            self.kalman_filters_blue = {}
            self.kalman_filter_ball = KalmanFilterBall()

    def start_filtering(self):
        """
        Start filtering after first valid frame is received from GameGater.
        """
        self._filter_running = True

    def _include_vanished_robots(
        self, vision_data: VisionData, game_frame: GameFrame
    ) -> Tuple[dict[int, Optional[VisionRobotData]], dict[int, Optional[VisionRobotData]]]:
        """
        Augment the VisionData lists with None for vanished robots so that the Kalman filter
        knows that data vanished.

        Returns:
            Tuple of (yellow_vision_dict, blue_vision_dict) where vanished robots are represented as None.
        """

        # TODO: major issue is that if we do a robot substitution, the
        # Kalman filter will think the old robot vanished and a new robot appeared.
        # needs to be adjusted when referee system is in place.
        # see issue #107 on GitHub for more details.

        yellow_ids_last_frame, blue_ids_last_frame = map_friendly_enemy_to_colors(
            game_frame.my_team_is_yellow,
            game_frame.friendly_robots.keys(),
            game_frame.enemy_robots.keys(),
        )

        # Current vision IDs
        yellow_present = {r.id for r in vision_data.yellow_robots}
        blue_present = {r.id for r in vision_data.blue_robots}

        # Start with current measurements
        yellow_vision_dict: dict[int, Optional[VisionRobotData]] = {r.id: r for r in vision_data.yellow_robots}
        blue_vision_dict: dict[int, Optional[VisionRobotData]] = {r.id: r for r in vision_data.blue_robots}

        # Add None for vanished robots
        for robot_id in yellow_ids_last_frame - yellow_present:
            yellow_vision_dict[robot_id] = None
        for robot_id in blue_ids_last_frame - blue_present:
            blue_vision_dict[robot_id] = None

        return yellow_vision_dict, blue_vision_dict

    # Static methods
    @staticmethod
    def _combine_robot_vision_data(
        old_robot: Robot, robot_data: VisionRobotData, angle_smoother: AngleSmoother
    ) -> Robot:
        assert old_robot.id == robot_data.id
        new_x, new_y = robot_data.x, robot_data.y

        # Needs fixing the bounds are off oren becoming -3.9rad
        # # Smoothing
        # new_orientation = angle_smoother.smooth(
        #     old_robot.orientation, robot_data.orientation
        # )
        return replace(
            old_robot,
            id=robot_data.id,
            p=Vector2D(new_x, new_y),
            orientation=robot_data.orientation,
        )

    # Used at start of the game so assume robot does not have the ball
    # Also assume velocity and acceleration are zero
    @staticmethod
    def _robot_from_vision(robot_data: VisionRobotData, is_friendly: bool) -> Robot:
        return Robot(
            id=robot_data.id,
            is_friendly=is_friendly,
            has_ball=False,
            p=Vector2D(robot_data.x, robot_data.y),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=robot_data.orientation,
        )

    @staticmethod
    def _ball_from_vision(ball_data: VisionBallData) -> Ball:
        zv = Vector3D(0, 0, 0)
        return Ball(Vector3D(ball_data.x, ball_data.y, ball_data.z), zv, zv)

    @staticmethod
    def _get_most_confident_ball(balls: List[VisionBallData]) -> Ball:
        balls_by_confidence = sorted(balls, key=lambda ball: ball.confidence, reverse=True)
        if not balls_by_confidence:
            return None
        return PositionRefiner._ball_from_vision(balls_by_confidence[0])

    def _combine_single_team_positions(
        self,
        new_game_robots: Dict[int, Robot],
        vision_robots: List[VisionRobotData],
        friendly: bool,
    ) -> Dict[int, Robot]:
        for robot in vision_robots:
            if robot.id not in new_game_robots:
                # At the start of the game, we haven't seen anything yet, so just create a new robot
                new_game_robots[robot.id] = PositionRefiner._robot_from_vision(robot, is_friendly=friendly)
            else:
                # Update with smoothed data.
                new_game_robots[robot.id] = PositionRefiner._combine_robot_vision_data(
                    new_game_robots[robot.id], robot, self.angle_smoother
                )
        return new_game_robots

    def _combine_both_teams_game_vision_positions(
        self,
        game_frame: GameFrame,
        yellow_vision_robots: List[VisionRobotData],
        blue_vision_robots: List[VisionRobotData],
    ) -> Tuple[Dict[int, Robot], Dict[int, Robot]]:
        if game_frame.my_team_is_yellow:
            old_yellow_robots = game_frame.friendly_robots.copy()
            old_blue_robots = game_frame.enemy_robots.copy()
        else:
            old_yellow_robots = game_frame.enemy_robots.copy()
            old_blue_robots = game_frame.friendly_robots.copy()

        new_yellow_robots = self._combine_single_team_positions(
            old_yellow_robots,
            yellow_vision_robots,
            friendly=game_frame.my_team_is_yellow,
        )
        new_blue_robots = self._combine_single_team_positions(
            old_blue_robots,
            blue_vision_robots,
            friendly=not game_frame.my_team_is_yellow,
        )

        return new_yellow_robots, new_blue_robots

    @property
    def filter_running(self) -> bool:
        return self._filter_running


class CameraCombiner:
    def combine_cameras(self, frames: List[RawVisionData], bounds: VisionBounds) -> VisionData:
        """
        Combines the vision data from multiple cameras into a single coherent VisionData object.
        Also, removes any robot detections that are out of the specified bounds.
        Args:
            frames (List[RawVisionData]): A list of RawVisionData objects from different cameras.
            bounds (VisionBounds): The bounds within which to consider vision data for combination.
        Returns:
            VisionData: A combined VisionData object containing averaged robot positions and the most confident ball position.
        """

        ts = []
        # maps robot id to list of frames seen for that robot
        yellow_captured = defaultdict(list)
        blue_captured = defaultdict(list)
        balls_captured = defaultdict(list)

        # Each frame is from a different camera
        for frame_ind, frame in enumerate(frames):
            for yr in frame.yellow_robots:
                if self._bounds_check(yr.x, yr.y, bounds):
                    yellow_captured[yr.id].append(yr)

            for br in frame.blue_robots:
                if self._bounds_check(br.x, br.y, bounds):
                    blue_captured[br.id].append(br)

            for b in frame.balls:
                if self._bounds_check(b.x, b.y, bounds):
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
        tx, ty, tc = 0, 0, 0

        sum_orientation_x_component = 0.0
        sum_orientation_y_component = 0.0
        for r in rs:
            assert base_id == r.id
            tx += r.x
            ty += r.y
            sum_orientation_x_component += np.cos(r.orientation)
            sum_orientation_y_component += np.sin(r.orientation)
            tc += r.confidence

        avg_orientation_x = sum_orientation_x_component / len(rs)
        avg_orientation_y = sum_orientation_y_component / len(rs)
        avg_orientation = np.atan2(avg_orientation_y, avg_orientation_x)

        return VisionRobotData(base_id, tx / len(rs), ty / len(rs), avg_orientation)

    def _combine_balls_by_proximity(self, bs: Dict[int, List[RawBallData]]) -> List[VisionBallData]:
        combined_balls: List[VisionBallData] = []
        for ball_list in bs.values():
            for b in ball_list:
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

    def _bounds_check(self, x: float, y: float, bounds: VisionBounds) -> bool:
        return bounds.x_min <= x <= bounds.x_max and bounds.y_min <= y <= bounds.y_max

    @staticmethod
    def ball_merge_predicate(b1: RawBallData, b2: RawBallData) -> bool:
        return abs(b1.x - b2.x) + abs(b1.y - b2.y) < BALL_MERGE_THRESHOLD

    @staticmethod
    def ball_merge(b1: RawBallData, b2: RawBallData) -> RawBallData:
        nx = (b1.x + b2.x) / 2
        ny = (b1.y + b2.y) / 2
        nz = (b1.z + b2.z) / 2
        nc = max(b1.confidence, b2.confidence)
        return RawBallData(nx, ny, nz, nc)
