import math
from typing import List, Tuple

import numpy as np  # type: ignore

from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import (
    closest_point_on_segment,
    distance,
    distance_between_line_segments,
    distance_point_to_segment,
    find_intersection,
    rotate_vector,
)
from utama_core.motion_planning.src.fastpathplanning.config import (
    fastpathplanningconfig as config,
)
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv


class FastPathPlanner:
    def __init__(self, env: SSLStandardEnv):
        self._env = env
        self.config = config
        self.OBSTACLE_CLEARANCE = self.config.OBSTACLE_CLEARANCE
        self.LOOK_AHEAD_RANGE = self.config.LOOK_AHEAD_RANGE
        self.SUBGOAL_DISTANCE = self.config.SUBGOAL_DISTANCE
        self.MAXRECURSIONLENGTH = self.config.MAXRECURSION_LENGTH
        self.PROJECTEDFRAMES = self.config.PROJECTEDFRAMES
        self.PROJECTION_DISTANCE = self.config.PROJECTION_DISTANCE

        # Initialize collision cache dictionary
        self._collision_cache = {}

    def is_point_in_field(self, point, field_bounds: FieldBounds) -> bool:
        x, y = float(point[0]), float(point[1])
        min_x = min(field_bounds.top_left[0], field_bounds.bottom_right[0])
        max_x = max(field_bounds.top_left[0], field_bounds.bottom_right[0])
        min_y = min(field_bounds.top_left[1], field_bounds.bottom_right[1])
        max_y = max(field_bounds.top_left[1], field_bounds.bottom_right[1])
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _get_obstacles(
        self, game: Game, robot_id: int, our_pos: np.ndarray, field_bounds: FieldBounds
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compiles obstacles and draws projected velocity lines in Red.
        """
        friendly_obstacles = [robot for robot in game.friendly_robots.values() if robot.id != robot_id]
        robots = friendly_obstacles + list(game.enemy_robots.values())
        obstacle_list = []

        for r in robots:
            robot_pos = np.array([r.p.x, r.p.y])
            if distance(our_pos, robot_pos) < self.LOOK_AHEAD_RANGE:
                velocity = np.array([r.v.x, r.v.y])
                # Project the "Ghost Wall" based on current velocity
                point = robot_pos + velocity * (self.PROJECTEDFRAMES / CONTROL_FREQUENCY)
                obstacle_segment = (robot_pos, point)

                obstacle_list.append(obstacle_segment)

                # DRAWING: Show the projected velocity line in Red
                self._env.draw_line(obstacle_segment, color="Red")

        # Field bounds as obstacles (static, usually not drawn to keep screen clean)
        tl, br = np.array(field_bounds.top_left), np.array(field_bounds.bottom_right)
        tr = np.array([field_bounds.bottom_right[0], field_bounds.top_left[1]])
        bl = np.array([field_bounds.top_left[0], field_bounds.bottom_right[1]])

        obstacle_list.extend([(tl, tr), (tr, br), (br, bl), (bl, tl)])
        return obstacle_list

    def _find_subgoal(
        self,
        robot_pos: np.ndarray,
        target: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacles: List,
        subgoal_direction: int,
        multiple: int,
    ) -> np.ndarray:

        # Failsafe to prevent infinite loops if completely trapped
        if multiple > 10:
            return obstacle_pos

        direction = target - robot_pos
        perp_dir = rotate_vector(direction[0], direction[1], math.pi * (subgoal_direction + 0.5))
        unitvec = perp_dir / np.linalg.norm(perp_dir)
        subgoal = obstacle_pos + self.SUBGOAL_DISTANCE * unitvec * multiple

        for o in obstacles:
            # OPTIMIZATION: Removed np.isclose, ensuring strictly less-than for clearance
            if distance_point_to_segment(subgoal, o[0], o[1]) < self.OBSTACLE_CLEARANCE:
                return self._find_subgoal(
                    robot_pos,
                    target,
                    obstacle_pos,
                    obstacles,
                    subgoal_direction,
                    multiple + 1,
                )
        return subgoal

    def collides(self, segment: Tuple, obstacles: List):
        # OPTIMIZATION: Cache collision results (convert numpy arrays to tuples for hashability)
        seg_key = (tuple(segment[0]), tuple(segment[1]))
        if seg_key in self._collision_cache:
            return self._collision_cache[seg_key]

        closest_obstacle = None
        min_dist_to_robot = float("inf")

        for o in obstacles:
            # OPTIMIZATION: Removed double distance call
            dist_between_segs = distance_between_line_segments(o[0], o[1], segment[0], segment[1])

            if dist_between_segs < self.OBSTACLE_CLEARANCE:
                # We want the obstacle closest to the START of the segment (the robot)
                dist_to_robot = distance_point_to_segment(segment[0], o[0], o[1])
                if dist_to_robot < min_dist_to_robot:
                    min_dist_to_robot = dist_to_robot
                    closest_obstacle = o

        obstacle_pos = None
        if closest_obstacle is not None:
            obstacle_pos = find_intersection(segment, closest_obstacle)
            if obstacle_pos is None:
                # Fallback to closest physical point if lines don't strictly intersect
                dists = [
                    distance_point_to_segment(closest_obstacle[0], segment[0], segment[1]),
                    distance_point_to_segment(closest_obstacle[1], segment[0], segment[1]),
                ]
                point_c = closest_point_on_segment(segment[0], closest_obstacle[0], closest_obstacle[1])
                point_d = closest_point_on_segment(segment[1], closest_obstacle[0], closest_obstacle[1])

                dists.extend([distance(segment[0], point_c), distance(segment[1], point_d)])
                points = [closest_obstacle[0], closest_obstacle[1], point_c, point_d]
                obstacle_pos = points[dists.index(min(dists))]

        # Save to cache
        self._collision_cache[seg_key] = obstacle_pos
        return obstacle_pos

    def _trajectory_length(self, trajectory):
        return sum(distance(seg[0], seg[1]) for seg in trajectory)

    def check_segment(
        self,
        segment: Tuple[np.ndarray, np.ndarray],
        obstacles: List[Tuple[np.ndarray, np.ndarray]],
        recursion_length: int,
        target: np.ndarray,
        field_bounds: FieldBounds,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
        """
        Recursively checks a segment for collisions and generates subgoals with
        a hysteresis bias to prevent path-switching jitter (indecisiveness).
        """
        closest_obstacle = self.collides(segment, obstacles)
        segment_length = distance(segment[0], segment[1])

        # Base case: Path is clear or maximum detour complexity reached
        if closest_obstacle is None or recursion_length >= self.MAXRECURSIONLENGTH:
            return [segment], segment_length

        # Generate left and right detours
        subgoal_left = self._find_subgoal(segment[0], segment[1], closest_obstacle, obstacles, 1, 1)
        subgoal_right = self._find_subgoal(segment[0], segment[1], closest_obstacle, obstacles, 0, 1)

        left_valid = self.is_point_in_field(subgoal_left, field_bounds)
        right_valid = self.is_point_in_field(subgoal_right, field_bounds)

        best_subgoal = None

        # Heuristic: Pick the valid subgoal closest to the ultimate destination
        if left_valid and right_valid:
            if distance(subgoal_left, target) < distance(subgoal_right, target):
                best_subgoal = subgoal_left
            else:
                best_subgoal = subgoal_right
        elif left_valid:
            best_subgoal = subgoal_left
        elif right_valid:
            best_subgoal = subgoal_right
        else:
            return [segment], segment_length

        # Recursively check the two halves of the selected detour
        seg1, len1 = self.check_segment(
            (segment[0], best_subgoal),
            obstacles,
            recursion_length + 1,
            target,
            field_bounds,
        )
        seg2, len2 = self.check_segment(
            (best_subgoal, segment[1]),
            obstacles,
            recursion_length + 1,
            target,
            field_bounds,
        )

        return seg1 + seg2, len1 + len2

    def smooth_path(self, trajectory, target, robot_position) -> np.ndarray:
        if len(trajectory) == 1:
            return target

        direction = trajectory[0][1] - robot_position
        unit_vec = direction / np.linalg.norm(direction)
        new_target = robot_position + unit_vec * self.PROJECTION_DISTANCE

        # Removed redundant math ops by caching distance calls here too
        dist_new_target = distance(new_target, robot_position)
        dist_trajectory = distance(robot_position, trajectory[0][1])

        if dist_new_target < dist_trajectory:
            return trajectory[0][1]
        else:
            point = closest_point_on_segment(new_target, trajectory[1][0], trajectory[1][1])
            return (point + new_target) / 2.0

    def sanitize_target(self, target: np.ndarray, obstacles: List, robot_pos: np.ndarray) -> np.ndarray:
        """
        Ensures the target isn't inside a velocity line or field bound.
        """
        safe_target = np.copy(target)
        for _ in range(5):
            collision_found = False
            for o in obstacles:
                if distance_point_to_segment(safe_target, o[0], o[1]) < self.OBSTACLE_CLEARANCE:
                    closest_pt = closest_point_on_segment(safe_target, o[0], o[1])
                    push_dir = safe_target - closest_pt
                    if np.linalg.norm(push_dir) == 0:
                        push_dir = robot_pos - closest_pt
                    unit_push = push_dir / np.linalg.norm(push_dir)
                    safe_target = closest_pt + unit_push * (self.OBSTACLE_CLEARANCE * 1.05)
                    collision_found = True
            if not collision_found:
                break
        return safe_target

    def _path_to(
        self,
        game: Game,
        robot_id: int,
        target: Tuple[float, float],
        field_bounds: FieldBounds,
    ):
        """
        Main entry point. Clears cache, sanitizes target, and plans path.
        """
        self._collision_cache.clear()

        robot = game.friendly_robots[robot_id]
        our_pos = np.array([robot.p.x, robot.p.y])
        raw_target = np.array(target)

        # 1. Get obstacles and draw Red velocity lines
        obstacles = self._get_obstacles(game, robot_id, our_pos, field_bounds)

        # 3. Sanitize target (Critical for velocity obstacles)
        safe_target = self.sanitize_target(raw_target, obstacles, our_pos)

        # 4. Plan geometric path
        final_trajectory, _ = self.check_segment((our_pos, safe_target), obstacles, 0, safe_target, field_bounds)

        # 5. Draw the resulting safe path segments
        for i in final_trajectory:
            self._env.draw_line(i)

        # 6. Smooth the path and draw the final "Carrot" target in Blue
        new_target = self.smooth_path(final_trajectory, safe_target, our_pos)
        self._env.draw_line((our_pos, new_target), color="Blue")

        return new_target
