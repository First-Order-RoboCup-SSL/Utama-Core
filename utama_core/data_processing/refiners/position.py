import math
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utama_core.config.field_params import FieldDimensions
from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.config.settings import (
    BALL_MERGE_THRESHOLD,
    BALL_ROBOT_OFFSET_MEMORY,
    BALL_ROBOT_SHIFT_RANGE,
    CAMERA_AGREEMENT,
    VISION_BOUNDS_BUFFER,
)
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


def _in_field_or_goal(x: float, y: float, dimensions: FieldDimensions) -> bool:
    if abs(y) <= dimensions.full_field_half_width and abs(x) <= dimensions.full_field_half_length:
        return True
    return abs(y) <= dimensions.half_goal_width and abs(x) <= dimensions.full_field_half_length + dimensions.goal_depth


@dataclass
class VisionBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class PositionRefiner(BaseRefiner):
    """
    Refiner that combines vision data from multiple cameras, applies bounds filtering,
    and optionally applies Kalman smoothing and/or imputes vanished robots.

    Args:
        full_field_dims: The dimensions of the full field, used to set bounds for vision data inclusion.
        filtering: Default for both `smooth_positions` and `impute_vanished` when they are left unset.
            Kept as a single convenience switch since most callers want both on or both off together.
        exp_ball: Whether to expect a ball on the field.
        smooth_positions: Whether to apply Kalman smoothing to robot/ball positions. Defaults to `filtering`.
        impute_vanished: Whether to impute robots that vanish from vision (via the Kalman predict step,
            with no measurement update that tick). Defaults to `filtering`. Note this is independent of
            smoothing: it can be kept on even if `smooth_positions` is off, or vice versa.

    Important:
        when exp_ball set to False, the refiner could return either Ball | None type ball value
        when exp_ball set to True, the refiner will ALWAYS return a Ball type ball value (will impute missing frames)
    """

    def __init__(
        self,
        full_field_dims: FieldDimensions,
        filtering: bool = True,
        exp_ball: bool = True,
        allowed_yellow_ids: Optional[frozenset[int]] = None,
        allowed_blue_ids: Optional[frozenset[int]] = None,
        smooth_positions: Optional[bool] = None,
        impute_vanished: Optional[bool] = None,
    ):
        top_left = full_field_dims.full_field_bounds.top_left
        bottom_right = full_field_dims.full_field_bounds.bottom_right

        self.vision_bounds = VisionBounds(
            x_min=top_left[0] - VISION_BOUNDS_BUFFER,
            x_max=bottom_right[0] + VISION_BOUNDS_BUFFER,
            y_min=bottom_right[1] - VISION_BOUNDS_BUFFER,
            y_max=top_left[1] + VISION_BOUNDS_BUFFER,
        )

        # Kalman smoothing and vanished-robot imputation are independent concerns that both
        # happen to be driven off the Kalman filter; each can be toggled on its own.
        self.smooth_positions = filtering if smooth_positions is None else smooth_positions
        self.impute_vanished = filtering if impute_vanished is None else impute_vanished
        self._uses_kalman = self.smooth_positions or self.impute_vanished
        self._filter_running = (
            False  # Only start filtering once we have valid data to filter (i.e. after the first valid game frame)
        )

        self._combiner = CameraCombiner(full_field_dims)
        self._field_dims = full_field_dims
        self.exp_ball = exp_ball
        self.allowed_yellow_ids = allowed_yellow_ids
        self.allowed_blue_ids = allowed_blue_ids

        if self._uses_kalman:
            # Instantiate a dedicated Kalman filter for each robot so filtering can be kept independent.
            self.kalman_filters_yellow: dict[int, KalmanFilter] = {}
            self.kalman_filters_blue: dict[int, KalmanFilter] = {}
            self.kalman_filter_ball = self._ball_filter()

    def _ball_filter(self) -> KalmanFilterBall:
        b = self.vision_bounds
        return KalmanFilterBall(bounds=(b.x_min, b.x_max, b.y_min, b.y_max))

    # Primary function for the Refiner interface
    def refine(
        self,
        game_frame: GameFrame,
        data: List[RawVisionData],
    ) -> GameFrame:
        frames = [
            replace(
                frame,
                yellow_robots=[
                    r for r in frame.yellow_robots if self.allowed_yellow_ids is None or r.id in self.allowed_yellow_ids
                ],
                blue_robots=[
                    r for r in frame.blue_robots if self.allowed_blue_ids is None or r.id in self.allowed_blue_ids
                ],
            )
            for frame in data
            if frame is not None
        ]

        # If no information just return the original
        if not frames:
            return game_frame

        combined_vision_data: VisionData = self._combiner.combine_cameras(
            frames,
            bounds=self.vision_bounds,
        )

        time_elapsed = combined_vision_data.ts - game_frame.ts

        # Kalman smoothing and vanished-robot imputation share the same predict/update filter,
        # so both toggles gate this block; each independently controls what it feeds into it.
        if self._uses_kalman and self._filter_running:  # Checks if the first valid game frame has been received.
            if self.impute_vanished:
                # Imputes combined_vision_data with null vision frames in place for vanished robots.
                vision_yellow, vision_blue = self._include_vanished_robots(combined_vision_data, game_frame)
            else:
                vision_yellow = {r.id: r for r in combined_vision_data.yellow_robots}
                vision_blue = {r.id: r for r in combined_vision_data.blue_robots}

            yellow_rbt_last_frame, blue_rbt_last_frame = map_friendly_enemy_to_colors(
                game_frame.my_team_is_yellow,
                game_frame.friendly_robots,
                game_frame.enemy_robots,
            )

            filtered_yellow_robots = []
            for y_rbt_id, vision_y_rbt in vision_yellow.items():
                if not self.smooth_positions and vision_y_rbt is not None:
                    # Smoothing off: pass present robots through untouched, still let vanished ones be imputed.
                    filtered_yellow_robots.append(vision_y_rbt)
                    continue
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
                if not self.smooth_positions and vision_b_rbt is not None:
                    filtered_blue_robots.append(vision_b_rbt)
                    continue
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

        new_ball = self._pick_ball(combined_vision_data.balls, game_frame.ball)

        # Skip filtering when there's no ball and we don't expect one
        if new_ball is not None or self.exp_ball:
            ball_vanished = new_ball is None
            run_ball_filter = self._filter_running and (
                (self.smooth_positions and not ball_vanished) or (self.impute_vanished and ball_vanished)
            )
            if run_ball_filter:
                new_ball = self.kalman_filter_ball.filter_data(
                    new_ball,
                    game_frame.ball,
                    time_elapsed,
                )
            elif new_ball is None:
                # If none, take the ball from the last frame of the game
                new_ball = game_frame.ball
            elif self._uses_kalman and self.kalman_filter_ball.state is not None:
                # Seen, unfiltered (smoothing off): forget the last loss, so the next one starts
                # from this ball rather than carrying on the old roll.
                self.kalman_filter_ball = self._ball_filter()

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
        self._combiner = CameraCombiner(self._field_dims)
        if self._uses_kalman:
            self.kalman_filters_yellow = {}
            self.kalman_filters_blue = {}
            self.kalman_filter_ball = self._ball_filter()

    def start_filtering(self):
        """
        Start filtering (imputation and interpolation) after first valid frame is received from GameGater.
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

        # Current vision IDs (filtered to roster if provided)
        yellow_robots = vision_data.yellow_robots
        blue_robots = vision_data.blue_robots
        if self.allowed_yellow_ids is not None:
            yellow_robots = [r for r in yellow_robots if r.id in self.allowed_yellow_ids]
            yellow_ids_last_frame = yellow_ids_last_frame & self.allowed_yellow_ids
        if self.allowed_blue_ids is not None:
            blue_robots = [r for r in blue_robots if r.id in self.allowed_blue_ids]
            blue_ids_last_frame = blue_ids_last_frame & self.allowed_blue_ids

        yellow_present = {r.id for r in yellow_robots}
        blue_present = {r.id for r in blue_robots}

        # Start with current measurements
        yellow_vision_dict: dict[int, Optional[VisionRobotData]] = {r.id: r for r in yellow_robots}
        blue_vision_dict: dict[int, Optional[VisionRobotData]] = {r.id: r for r in blue_robots}

        # Add None for vanished robots
        for robot_id in yellow_ids_last_frame - yellow_present:
            yellow_vision_dict[robot_id] = None
        for robot_id in blue_ids_last_frame - blue_present:
            blue_vision_dict[robot_id] = None

        return yellow_vision_dict, blue_vision_dict

    # Static methods
    @staticmethod
    def _combine_robot_vision_data(old_robot: Robot, robot_data: VisionRobotData) -> Robot:
        assert old_robot.id == robot_data.id
        new_x, new_y = robot_data.x, robot_data.y

        # Built directly rather than with `dataclasses.replace`: the same Robot, without
        # replace's per-call overhead (every robot, every frame).
        return Robot(
            robot_data.id,
            old_robot.is_friendly,
            old_robot.has_ball,
            Vector2D(new_x, new_y),
            old_robot.v,
            old_robot.a,
            robot_data.orientation,
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

    def _pick_ball(self, balls: List[VisionBallData], last: Optional[Ball]) -> Optional[Ball]:
        """The ball among `balls`: one inside the field lines or a goal when any is (a ball rarely
        leaves the field, stray orange things off it do: 2026-10-22, four steady "balls" 15 cm
        past the touch line, as confident as the real one), then the nearest to `last`, the ball
        the frame before. With no last ball, the most confident."""
        inside = [b for b in balls if _in_field_or_goal(b.x, b.y, self._field_dims)]
        candidates = inside or balls
        if last is None or not candidates:
            return PositionRefiner._get_most_confident_ball(candidates)
        nearest = min(candidates, key=lambda b: math.hypot(b.x - last.p.x, b.y - last.p.y))
        return PositionRefiner._ball_from_vision(nearest)

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
                new_game_robots[robot.id] = PositionRefiner._combine_robot_vision_data(new_game_robots[robot.id], robot)
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

        if self.allowed_yellow_ids is not None:
            yellow_vision_robots = [r for r in yellow_vision_robots if r.id in self.allowed_yellow_ids]
            old_yellow_robots = {k: v for k, v in old_yellow_robots.items() if k in self.allowed_yellow_ids}
        if self.allowed_blue_ids is not None:
            blue_vision_robots = [r for r in blue_vision_robots if r.id in self.allowed_blue_ids]
            old_blue_robots = {k: v for k, v in old_blue_robots.items() if k in self.allowed_blue_ids}

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
    """Merges the cameras' latest frames into one. A robot seen by several cameras is placed at
    the average of their detections, each weighted by how steeply its camera looks at the robot
    when ssl-vision sends where the cameras are: a camera's error on the floor grows as
    1/cos^2 of its view angle (a pixel covers more floor, a wrong robot height shifts the robot
    more), so the weight is cos^4. A detection further than `CAMERA_AGREEMENT` from the
    best-placed camera's is dropped, so a camera that disagrees (a poor calibration far from it)
    does not pull the robot away. Without camera positions every camera counts the same. A ball
    is fused by the same rule, its detections grouped by being within `BALL_MERGE_THRESHOLD`
    of every other member: fused one way and the robots another, a ball at a robot's dribbler came out
    inside the robot (2026-10-22). Each camera's ball is first moved by that camera's offset at
    the nearest robot it also sees (within `BALL_ROBOT_SHIFT_RANGE`): the fused robot minus its
    own detection of it. So a ball seen by one camera only (the robot hides it from the other)
    stays where that camera put it relative to the robot. A camera that has just lost the robot
    (the ball at its dribbler still in view) puts it where another camera sees it plus the gap
    between the two cameras there in the last `BALL_ROBOT_OFFSET_MEMORY` s: the gap is their
    calibrations' disagreement, steady over that time. Without it the ball came out inside the
    robot (2.4% of one 2026-10-22 log)."""

    def __init__(self, full_field_dims: Optional[FieldDimensions] = None):
        self._field_dims = full_field_dims
        # (camera id, other camera id, (colour, robot id)) -> (camera's detection minus the other's
        # in x, y; frame time)
        self._gaps: Dict[tuple, Tuple[float, float, float]] = {}
        # (colour, robot id) -> (fused robot, {camera id: (x, y)}, frame time)
        self._remembered: Dict[Tuple[str, int], Tuple[VisionRobotData, Dict[int, Tuple[float, float]], float]] = {}
        # (camera id, (colour, robot id)) -> (x, y, frame time, camera position)
        self._last_seen: Dict[tuple, Tuple[float, float, float, Optional[Tuple[float, float, float]]]] = {}

    def combine_cameras(
        self,
        frames: List[RawVisionData],
        bounds: VisionBounds,
    ) -> VisionData:
        """
        Combines the vision data from multiple cameras into a single coherent VisionData object.
        Also, removes any robot detections that are out of the specified bounds.
        Args:
            frames (List[RawVisionData]): A list of RawVisionData objects from different cameras.
            bounds (VisionBounds): The bounds within which to consider vision data for combination.
        Returns:
            VisionData: A combined VisionData object containing averaged robot positions and the most confident ball position.
        """

        frames = [
            replace(
                frame,
                yellow_robots=self._nearby_robot_duplicates(
                    [r for r in frame.yellow_robots if self._bounds_check(r.x, r.y, bounds)]
                ),
                blue_robots=self._nearby_robot_duplicates(
                    [r for r in frame.blue_robots if self._bounds_check(r.x, r.y, bounds)]
                ),
            )
            for frame in frames
        ]
        ts = []
        # maps robot id to the (detection, its camera's position) of every camera that saw it
        yellow_captured = defaultdict(list)
        blue_captured = defaultdict(list)

        # Each frame is from a different camera
        for frame in frames:
            for yr in frame.yellow_robots:
                if self._bounds_check(yr.x, yr.y, bounds):
                    yellow_captured[yr.id].append((yr, frame.camera_position))

            for br in frame.blue_robots:
                if self._bounds_check(br.x, br.y, bounds):
                    blue_captured[br.id].append((br, frame.camera_position))
            ts.append(frame.ts)

        frame_ts = sum(ts) / len(ts)
        avg_yellows = list(map(self._avg_robots, yellow_captured.values()))
        avg_blues = list(map(self._avg_robots, blue_captured.values()))
        fused = {("yellow", r.id): r for r in avg_yellows} | {("blue", r.id): r for r in avg_blues}

        # Each ball in the fused robots' frame: shifted by its camera's offset at its nearest robot.
        seen_by = defaultdict(dict)  # (colour, robot id) -> camera id -> that camera's detection
        for frame in frames:
            for colour, robots in (("yellow", frame.yellow_robots), ("blue", frame.blue_robots)):
                for r in robots:
                    if self._bounds_check(r.x, r.y, bounds):
                        seen_by[(colour, r.id)][frame.camera_id] = (r.x, r.y)

        # Keep single-camera ticks in the fused frame using the last multi-camera bias.
        for key, detections in seen_by.items():
            f = fused[key]
            if len(detections) >= 2:
                self._remembered[key] = (f, dict(detections), frame_ts)
            elif len(detections) == 1:
                cam_id, (x, y) = next(iter(detections.items()))
                rem = self._remembered.get(key)
                if rem is None or frame_ts - rem[2] > BALL_ROBOT_OFFSET_MEMORY:
                    for (other, last_key), (ox, oy, ot, o_pos) in list(self._last_seen.items()):
                        if last_key == key and other != cam_id and abs(frame_ts - ot) <= 0.05:
                            captured = yellow_captured[key[1]] if key[0] == "yellow" else blue_captured[key[1]]
                            f = self._avg_robots(captured + [(RawRobotData(key[1], ox, oy, f.orientation, 1.0), o_pos)])
                            self._remembered[key] = (f, {cam_id: (x, y), other: (ox, oy)}, frame_ts)
                            self._gaps[(cam_id, other, key)] = (x - ox, y - oy, frame_ts)
                            self._gaps[(other, cam_id, key)] = (ox - x, oy - y, frame_ts)
                            rem = self._remembered[key]
                            break
                if rem is not None and frame_ts - rem[2] <= BALL_ROBOT_OFFSET_MEMORY and cam_id in rem[1]:
                    rx, ry = rem[1][cam_id]
                    fused[key] = VisionRobotData(f.id, x + rem[0].x - rx, y + rem[0].y - ry, f.orientation)

        for frame in frames:
            for colour, robots in (("yellow", frame.yellow_robots), ("blue", frame.blue_robots)):
                for r in robots:
                    self._last_seen[(frame.camera_id, (colour, r.id))] = (r.x, r.y, frame.ts, frame.camera_position)

        avg_yellows = [fused[("yellow", r.id)] for r in avg_yellows]
        avg_blues = [fused[("blue", r.id)] for r in avg_blues]

        shifted = []
        all_keys = set(fused) | {
            k for k, rem in self._remembered.items() if frame_ts - rem[2] <= BALL_ROBOT_OFFSET_MEMORY
        }
        for frame in frames:
            own = []  # (where this camera puts a robot, the fused robot there)
            for key in all_keys:
                f = fused.get(key) or self._remembered[key][0]
                detections = seen_by.get(key, {})
                if frame.camera_id in detections:
                    x, y = detections[frame.camera_id]
                    for other, (ox, oy) in detections.items():
                        self._gaps[(frame.camera_id, other, key)] = (x - ox, y - oy, frame.ts)
                    own.append((x, y, f))
                    continue
                # Lost for a moment: where it would be, from another camera and the gap between them.
                for other, (ox, oy) in detections.items():
                    gap = self._gaps.get((frame.camera_id, other, key))
                    if gap is not None and frame.ts - gap[2] <= BALL_ROBOT_OFFSET_MEMORY:
                        own.append((ox + gap[0], oy + gap[1], f))
                        break
                else:
                    rem = self._remembered.get(key)
                    if rem is not None and frame.ts - rem[2] <= BALL_ROBOT_OFFSET_MEMORY and frame.camera_id in rem[1]:
                        rx, ry = rem[1][frame.camera_id]
                        own.append((f.x - (rem[0].x - rx), f.y - (rem[0].y - ry), f))
            for b in frame.balls:
                if not self._bounds_check(b.x, b.y, bounds):
                    continue
                near = min(own, key=lambda o: math.hypot(o[0] - b.x, o[1] - b.y), default=None)
                if near is not None and math.hypot(near[0] - b.x, near[1] - b.y) <= BALL_ROBOT_SHIFT_RANGE:
                    x, y, f = near
                    b = RawBallData(b.x + f.x - x, b.y + f.y - y, b.z, b.confidence)
                shifted.append((b, frame.camera_position, frame.camera_id))
        balls = self._combine_balls_by_proximity(shifted, avg_yellows + avg_blues)

        return VisionData(frame_ts, avg_yellows, avg_blues, balls)

    @staticmethod
    def _nearby_robot_duplicates(robots: List[RawRobotData]) -> List[RawRobotData]:
        """Nearby repeats of one ID in an image are one camera observation."""
        by_id = defaultdict(list)
        for robot in robots:
            by_id[robot.id].append(robot)
        result = []
        for robot_id, seen in by_id.items():
            if len(seen) < 2 or any(math.hypot(a.x - b.x, a.y - b.y) > CAMERA_AGREEMENT for a in seen for b in seen):
                result.extend(seen)
                continue
            n = len(seen)
            result.append(
                RawRobotData(
                    robot_id,
                    sum(r.x for r in seen) / n,
                    sum(r.y for r in seen) / n,
                    math.atan2(sum(math.sin(r.orientation) for r in seen), sum(math.cos(r.orientation) for r in seen)),
                    max(r.confidence for r in seen),
                )
            )
        return result

    @staticmethod
    def _view_weight(x: float, y: float, camera: Optional[Tuple[float, float, float]]) -> float:
        """cos^4 of the angle from the vertical at which `camera` sees (`x`, `y`); 1 when unknown."""
        if camera is None:
            return 1.0
        cx, cy, height = camera
        h2 = height * height
        return (h2 / (h2 + (x - cx) ** 2 + (y - cy) ** 2)) ** 2

    def _fuse(self, seen: list) -> list:
        """(detection, weight) of the detections of one thing, `seen` as (detection, camera
        position): those within `CAMERA_AGREEMENT` of the best-placed camera's, weighted by view
        angle; every detection at weight 1 when a camera's position is unknown."""
        if any(camera is None for _, camera in seen):
            return [(d, 1.0) for d, _ in seen]
        weights = [self._view_weight(d.x, d.y, camera) for d, camera in seen]
        best = seen[int(np.argmax(weights))][0]
        return [
            (d, w) for (d, _), w in zip(seen, weights) if math.hypot(d.x - best.x, d.y - best.y) <= CAMERA_AGREEMENT
        ]

    def _avg_robots(self, seen: List[Tuple[RawRobotData, Optional[Tuple[float, float, float]]]]) -> VisionRobotData:
        # All these robots should have the same id
        base_id = seen[0][0].id
        assert all(r.id == base_id for r, _ in seen)
        kept = self._fuse(seen)
        total = sum(w for _, w in kept)
        x = sum(r.x * w for r, w in kept) / total
        y = sum(r.y * w for r, w in kept) / total
        avg_orientation = math.atan2(
            sum(math.sin(r.orientation) * w for r, w in kept),
            sum(math.cos(r.orientation) * w for r, w in kept),
        )
        return VisionRobotData(base_id, x, y, avg_orientation)

    def _plausible_ball(self, ball: RawBallData, robots: Sequence[VisionRobotData]) -> bool:
        in_field = self._field_dims is None or _in_field_or_goal(ball.x, ball.y, self._field_dims)
        clear_of_robots = ball.z > BALL_RADIUS or all(
            math.hypot(ball.x - robot.x, ball.y - robot.y) >= ROBOT_RADIUS for robot in robots
        )
        return in_field and clear_of_robots

    def _combine_balls_by_proximity(
        self,
        seen: list[tuple[RawBallData, Optional[tuple[float, float, float]], int]],
        robots: Sequence[VisionRobotData] = (),
    ) -> List[VisionBallData]:
        """Fuse close observations, using the strongest candidate per camera in each group."""
        groups: list = []
        for detection in seen:
            ball = detection[0]
            for group in groups:
                if all(math.hypot(ball.x - b.x, ball.y - b.y) < BALL_MERGE_THRESHOLD for b, _, _ in group):
                    group.append(detection)
                    break
            else:
                groups.append([detection])
        balls = []
        for group in groups:
            # Prefer a plausible nearby alternative before confidence can discard it.
            # If every candidate is uncertain, retain the existing measurement path.
            candidates = [observation for observation in group if self._plausible_ball(observation[0], robots)] or group
            by_camera = {}
            for ball, camera, camera_id in candidates:
                previous = by_camera.get(camera_id)
                if previous is None or ball.confidence > previous[0].confidence:
                    by_camera[camera_id] = (ball, camera)
            kept = self._fuse(list(by_camera.values()))
            total = sum(w for _, w in kept)
            balls.append(
                RawBallData(
                    sum(b.x * w for b, w in kept) / total,
                    sum(b.y * w for b, w in kept) / total,
                    sum(b.z * w for b, w in kept) / total,
                    max(b.confidence for b, _ in kept),
                )
            )
        return balls

    def _bounds_check(self, x: float, y: float, bounds: VisionBounds) -> bool:
        return bounds.x_min <= x <= bounds.x_max and bounds.y_min <= y <= bounds.y_max
