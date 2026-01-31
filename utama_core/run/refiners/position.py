from collections import defaultdict
import csv
from dataclasses import replace
from functools import partial
import numpy as np
from typing import Dict, List, Optional, Tuple

from utama_core.config.settings import BALL_MERGE_THRESHOLD
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.data.vision import VisionBallData, VisionData, VisionRobotData
from utama_core.entities.game import Ball, FieldBounds, GameFrame, Robot
from utama_core.run.refiners.base_refiner import BaseRefiner
from utama_core.run.refiners.kalman import Kalman_filter, Kalman_filter_ball

from numpy.random import normal
from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise
from utama_core.global_utils.math_utils import normalise_heading_deg, deg_to_rad

# For logging
TS_COL, ID_COL, COLOR_COL = "ts", "id", "color"
X_COL, Y_COL, Z_COL, TH_COL = "x", "y", "z", "orientation"
COLS = [TS_COL, ID_COL, COLOR_COL, X_COL, Y_COL, TH_COL]
BALL_COLS = [TS_COL, X_COL, Y_COL, Z_COL]

OUTPUT_FILE   = "clean-dwa-pc-r-3.csv"
OUTPUT_FILE_2 = "noisy-10-dwa-pc-r-3.csv"
OUTPUT_FILE_3 = "noisy-10-kalman-dwa-pc-r-3.csv"
BALL_FILE     = "clean-dwa-pc-r-ball.csv"
BALL_FILE_2   = "noisy-10-dwa-pc-r-ball.csv"
BALL_FILE_3   = "noisy-10-kalman-dwa-pc-r-ball.csv"
TARGET_SIZE   = 8000


class AngleSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor for angle
        self.smoothed_angles = {}  # Stores last smoothed angle for each robot

    def smooth(self, old_angle: float, new_angle: float) -> float:
        # Compute the shortest angular difference
        diff = np.atan2(np.sin(new_angle - old_angle), np.cos(new_angle - old_angle))
        smoothed_angle = old_angle + self.alpha * diff

        return smoothed_angle


class PositionRefiner(BaseRefiner):    
    def __init__(
        self,
        my_team_is_yellow: bool,
        exp_friendly: int,
        exp_enemy: int,
        field_bounds: FieldBounds,
        bounds_buffer: float = 1.0
    ):
        # alpha=0 means no change in angle (inf smoothing), alpha=1 means no smoothing
        self.angle_smoother = AngleSmoother(alpha=1)
        self.x_min = field_bounds.top_left[0] - bounds_buffer  # expand left
        self.x_max = field_bounds.bottom_right[0] + bounds_buffer  # expand right
        self.y_min = field_bounds.bottom_right[1] - bounds_buffer  # expand bottom
        self.y_max = field_bounds.top_left[1] + bounds_buffer  # expand top
        self.BOUNDS_BUFFER = bounds_buffer
        
        # For Kalman filtering
        if my_team_is_yellow:
            self.yellow_count = exp_friendly
            self.blue_count = exp_enemy
        else:
            self.yellow_count = exp_enemy
            self.blue_count = exp_friendly
        
        self.yellow_range = set(range(self.yellow_count))
        self.blue_range = set(range(self.blue_count))
        
        # Instantiate a dedicated Kalman filter for each robot so filtering can be kept independent.
        self.kalman_filters_yellow = [Kalman_filter(id) for id in self.yellow_range]
        self.kalman_filters_blue   = [Kalman_filter(id) for id in self.blue_range]
        self.kalman_filter_ball    = Kalman_filter_ball()
        
        # class GameFrame: ts: float, my_team_is_yellow: bool, my_team_is_right: bool
        # friendly_robots: Dict[int, Robot], enemy_robots: Dict[int, Robot], ball: Optional[Ball]
        # class Robot: id: int, is_friendly: bool, has_ball: bool, p: Vector2D, v: Vector2D, a: Vector2D, orientation: float
        self.running = False  # Imputing is only turned on when the refiner is run from _step_game(), not _load_game()
        self.last_game_frame = None  # Game gater will initialise
        
        # For logging
        self.data_collected = 0
        
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, COLS)
            writer.writeheader()
            
        with open(OUTPUT_FILE_2, "w", newline="") as f:
            writer = csv.DictWriter(f, COLS)
            writer.writeheader()
            
        with open(OUTPUT_FILE_3, "w", newline="") as f:
            writer = csv.DictWriter(f, COLS)
            writer.writeheader()
        
        with open(BALL_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, BALL_COLS)
            writer.writeheader()
            
        with open(BALL_FILE_2, "w", newline="") as f:
            writer = csv.DictWriter(f, BALL_COLS)
            writer.writeheader()
            
        with open(BALL_FILE_3, "w", newline="") as f:
            writer = csv.DictWriter(f, BALL_COLS)
            writer.writeheader()

    # Primary function for the Refiner interface
    def refine(self, game_frame: GameFrame, data: List[RawVisionData]) -> GameFrame:
        frames = [frame for frame in data if frame is not None]

        # If no information just return the original
        # TODO: this needs to be replaced by an extrapolation function (otherwise we will be using old data forever)
        if not frames:
            return game_frame
        
        # class VisionData: ts: float; yellow_robots: List[VisionRobotData]; blue_robots: List[VisionRobotData]; balls: List[VisionBallData]
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        combined_vision_data: VisionData = CameraCombiner().combine_cameras(frames)
            
        if self.running:  # Checks if the first valid game frame has been received.
            # For logging:
            if self.data_collected < TARGET_SIZE:
                with open(OUTPUT_FILE, "a", newline="") as f:
                    writer = csv.DictWriter(f, COLS)
                    
                    for y_robot in sorted(combined_vision_data.yellow_robots, key=lambda r: r.id):
                        writer.writerow({
                            TS_COL: combined_vision_data.ts,
                            ID_COL: y_robot.id,
                            COLOR_COL: "yellow",
                            X_COL: y_robot.x,
                            Y_COL: y_robot.y,
                            TH_COL: y_robot.orientation
                        })
                        
            for robot in combined_vision_data.yellow_robots:
                PositionRefiner._add_gaussian_noise_robot(robot, x_stddev=0, y_stddev=0, th_stddev_deg=1)
            
            # For logging:
            if self.data_collected < TARGET_SIZE:
                with open(OUTPUT_FILE_2, "a", newline="") as f:
                    writer = csv.DictWriter(f, COLS)
                    
                    for y_robot in sorted(combined_vision_data.yellow_robots, key=lambda r: r.id):
                        writer.writerow({
                            TS_COL: combined_vision_data.ts,
                            ID_COL: y_robot.id,
                            COLOR_COL: "yellow",
                            X_COL: y_robot.x,
                            Y_COL: y_robot.y,
                            TH_COL: y_robot.orientation
                        })
        
            # For vanishing: imputes combined_vision_data with null vision frames in place.
            self._impute_vanished(combined_vision_data)
            
            # # For filtering
            time_elapsed = combined_vision_data.ts - self.last_game_frame.ts
            
            if game_frame.my_team_is_yellow:
                yellow_last = self.last_game_frame.friendly_robots
                blue_last = self.last_game_frame.enemy_robots
            else:
                yellow_last = self.last_game_frame.enemy_robot
                blue_last = self.last_game_frame.friendly_robots
                
            combined_vision_data = VisionData(
                ts=combined_vision_data.ts,
                
                yellow_robots = list(
                    map(partial(Kalman_filter.filter_data,
                                last_frame=yellow_last,
                                time_elapsed=time_elapsed),
                    self.kalman_filters_yellow,
                    sorted(combined_vision_data.yellow_robots, key=lambda r: r.id))
                    ),
                
                blue_robots = list(
                    map(partial(Kalman_filter.filter_data,
                                last_frame=blue_last,
                                time_elapsed=time_elapsed),
                    self.kalman_filters_blue,
                    sorted(combined_vision_data.blue_robots, key=lambda r: r.id))
                    ),
                
                balls=combined_vision_data.balls
            )
            
            # For logging:
            if self.data_collected < TARGET_SIZE:
                with open(OUTPUT_FILE_3, "a", newline="") as f:
                    writer = csv.DictWriter(f, COLS)
                    
                    for y_robot in sorted(combined_vision_data.yellow_robots, key=lambda r: r.id):
                        writer.writerow({
                            TS_COL: combined_vision_data.ts,
                            ID_COL: y_robot.id,
                            COLOR_COL: "yellow",
                            X_COL: y_robot.x,
                            Y_COL: y_robot.y,
                            TH_COL: y_robot.orientation
                        })
                        
                    # for b_robot in combined_vision_data.blue_robots:
                    #     writer.writerow({
                    #         TS_COL: combined_vision_data.ts,
                    #         ID_COL: b_robot.id,
                    #         COLOR_COL: "blue",
                    #         X_COL: b_robot.x,
                    #         Y_COL: b_robot.y,
                    #         TH_COL: b_robot.orientation
                    #     })

        # Some processing of robot vision data
        new_yellow_robots, new_blue_robots = self._combine_both_teams_game_vision_positions(
            game_frame,
            combined_vision_data.yellow_robots,
            combined_vision_data.blue_robots,
        )

        # After the balls have been combined, take the most confident
        new_ball: Ball = PositionRefiner._get_most_confident_ball(combined_vision_data.balls)
        if self.running:
            # For logging:
            if self.data_collected < TARGET_SIZE:
                with open(BALL_FILE, "a", newline="") as f:
                    writer = csv.DictWriter(f, BALL_COLS)
                    
                    writer.writerow({
                        TS_COL: combined_vision_data.ts,
                        X_COL: new_ball.p.x,
                        Y_COL: new_ball.p.y,
                        Z_COL: new_ball.p.z
                    })
                    
            PositionRefiner._add_gaussian_noise_ball(new_ball, x_stddev=0, y_stddev=0)
            
            if self.data_collected < TARGET_SIZE:
                with open(BALL_FILE_2, "a", newline="") as f:
                    writer = csv.DictWriter(f, BALL_COLS)
                    
                    writer.writerow({
                        TS_COL: combined_vision_data.ts,
                        X_COL: new_ball.p.x,
                        Y_COL: new_ball.p.y,
                        Z_COL: new_ball.p.z
                    })
            
            new_ball = Kalman_filter_ball.filter_data(self.kalman_filter_ball, new_ball, self.last_game_frame.ball, time_elapsed)
            
            if self.data_collected < TARGET_SIZE:
                with open(BALL_FILE_3, "a", newline="") as f:
                    writer = csv.DictWriter(f, BALL_COLS)
                    
                    writer.writerow({
                        TS_COL: combined_vision_data.ts,
                        X_COL: new_ball.p.x,
                        Y_COL: new_ball.p.y,
                        Z_COL: new_ball.p.z
                    })
                    
            self.data_collected += 1
            
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
                
    
    def _impute_vanished(
        self,
        vision_data: VisionData
    ) -> None:  # Imputes in place
        """
        Just to impute a placeholder, so that the Kalman filter knows that data
        vanished.

        Args:
            vision_data (VisionData): The vision data with missing robots to be
                imputed in place
        """
        
        yellows_present = { robot.id for robot in vision_data.yellow_robots }
        yellows_vanished = self.yellow_range - yellows_present
            
        for robot_id in yellows_vanished:
            vision_data.yellow_robots.append(
                VisionRobotData(id=robot_id, x=None, y=None, orientation=None)
            )                            
        
        blues_present = { robot.id for robot in vision_data.blue_robots }
        blues_vanished = self.blue_range - blues_present
            
        for robot_id in blues_vanished:
            vision_data.blue_robots.append(
                VisionRobotData(id=robot_id, x=None, y=None, orientation=None)
            )
    
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
            new_x, new_y = robot.x, robot.y

            if not (self.x_min <= new_x <= self.x_max and self.y_min <= new_y <= self.y_max):
                # Out of bounds so ignore this robot
                continue

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


    @staticmethod
    def _add_gaussian_noise_ball(ball: Ball, x_stddev: float=0.1, y_stddev: float=0.1):
        """
        When running in rsim, add Gaussian noise to ball with the given standard deviations.
        Mutates the Robot object in place.
        
        Args:
            noise (RsimGaussianNoise): The 3 parameters are for x (in m), y (in m), and orientation (in degrees) respectively.
                Defaults to 0 for each.
        """
        
        if x_stddev:
            ball.p.x += normal(scale=x_stddev)
            
        if y_stddev:
            ball.p.y += normal(scale=y_stddev)
            
        # No noise addition for z, since rSim is 2-D
        
    @staticmethod
    def _add_gaussian_noise_robot(robot: VisionRobotData, x_stddev: float=0.1, y_stddev: float=0.1, th_stddev_deg: float=5):
        """
        When running in rsim, add Gaussian noise to robot with the given standard deviations.
        Mutates the Robot object in place.
        
        Args:
            noise (RsimGaussianNoise): The 3 parameters are for x (in m), y (in m), and orientation (in degrees) respectively.
                Defaults to 0 for each.
        """
        
        if x_stddev:
            robot.x += normal(scale=x_stddev)
            
        if y_stddev:
            robot.y += normal(scale=y_stddev)
            
        if th_stddev_deg:
            robot.orientation = normalise_heading_deg(robot.orientation + normal(scale=deg_to_rad(th_stddev_deg)))


class CameraCombiner:
    def combine_cameras(self, frames: List[RawVisionData]) -> VisionData:
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