import math
from typing import List, Tuple

import numpy as np  # type: ignore

from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import (
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

    def is_point_in_field(self, point, field_bounds: FieldBounds) -> bool:
        x = float(point[0])
        y = float(point[1])

        min_x = min(field_bounds.top_left[0], field_bounds.bottom_right[0])
        max_x = max(field_bounds.top_left[0], field_bounds.bottom_right[0])
        min_y = min(field_bounds.top_left[1], field_bounds.bottom_right[1])
        max_y = max(field_bounds.top_left[1], field_bounds.bottom_right[1])

        return min_x <= x <= max_x and min_y <= y <= max_y

    def _get_obstacles(self, game: Game, robot_id: int, our_pos, field_bounds: FieldBounds):
        friendly_obstacles = [robot for robot in game.friendly_robots.values() if robot.id != robot_id]
        robots = friendly_obstacles + list(game.enemy_robots.values())
        obstacle_list = []
        for r in robots:
            robot_pos = np.array([r.p.x, r.p.y])
            if distance(our_pos, robot_pos) < self.LOOK_AHEAD_RANGE:
                velocity = np.array([r.v.x, r.v.y])
                point = np.array([r.p.x, r.p.y]) + velocity * self.PROJECTEDFRAMES / 60
                obstalce_segment = (robot_pos, point)
                obstacle_list.append(obstalce_segment)
        obstacle_list.append(
            (np.array(field_bounds.top_left), np.array([field_bounds.bottom_right[0], field_bounds.top_left[1]]))
        )
        obstacle_list.append(
            (np.array([field_bounds.bottom_right[0], field_bounds.top_left[1]]), np.array(field_bounds.bottom_right))
        )
        obstacle_list.append(
            (np.array(field_bounds.bottom_right), np.array([field_bounds.top_left[0], field_bounds.bottom_right[1]]))
        )
        obstacle_list.append(
            (np.array([field_bounds.top_left[0], field_bounds.bottom_right[1]]), field_bounds.top_left)
        )
        return obstacle_list

    def _find_subgoal(
        self,
        robot_pos: np.array,
        target: np.array,
        obstacle_pos: np.array,
        obstacles: List,
        subgoal_direction: int,
        multiple: int,
    ) -> np.array:

        direction = target - robot_pos
        perp_dir = rotate_vector(direction[0], direction[1], math.pi * (subgoal_direction + 1 / 2))
        unitvec = perp_dir / np.linalg.norm(perp_dir)
        subgoal = obstacle_pos + self.SUBGOAL_DISTANCE * unitvec * multiple

        for o in obstacles:
            if np.isclose(distance_point_to_segment(subgoal, o[0], o[1]), self.OBSTACLE_CLEARANCE):
                subgoal = self._find_subgoal(
                    robot_pos,
                    target,
                    obstacle_pos,
                    obstacles,
                    subgoal_direction,
                    multiple + 1,
                )
        return subgoal

    def collides(
        self,
        segment: Tuple,
        obstacles: List,
    ):  # returns None if no obstacles, else it returns the closest obstacle.
        closest_obstacle = None
        obstacle_pos = None
        tempdistance = distance(segment[0], segment[1])
        for o in obstacles:

            if distance_between_line_segments(o[0], o[1], segment[0], segment[1]) < self.OBSTACLE_CLEARANCE:
                obstacledistance = distance_between_line_segments(o[0], o[1], segment[0], segment[1])
                if closest_obstacle is None or obstacledistance < tempdistance:
                    tempdistance = obstacledistance
                    closest_obstacle = o
        if closest_obstacle is not None:
            obstacle_pos = find_intersection(segment, closest_obstacle)
            if obstacle_pos is None:
                if distance_point_to_segment(closest_obstacle[0], segment[0], segment[1]) < distance_point_to_segment(
                    closest_obstacle[1], segment[0], segment[1]
                ):
                    obstacle_pos = closest_obstacle[0]
                else:
                    obstacle_pos = closest_obstacle[1]
        return obstacle_pos

    def _trajectory_length(self, trajectory):
        trajectory_legnth = 0
        for i in trajectory:
            trajectory_legnth += distance(i[0], i[1])
        return trajectory_legnth

    def checksegment(
        self, segment: Tuple, obstacles: List, recursionlength: int, target: np.array, field_bounds: FieldBounds
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
        """
        If there are obstacles in the segment, divide the segment and return subsegments
        along with the total trajectory length.
        """

        closestobstacle = self.collides(segment, obstacles)
        if closestobstacle is not None and recursionlength < self.MAXRECURSIONLENGTH:
            # compute left and right subgoals explicitly
            subgoal_left = self._find_subgoal(
                segment[0],
                segment[1],
                closestobstacle,
                obstacles,
                subgoal_direction=1,
                multiple=1,
            )
            subgoal_right = self._find_subgoal(
                segment[0],
                segment[1],
                closestobstacle,
                obstacles,
                subgoal_direction=0,
                multiple=1,
            )

            # recursively check subsegments for left side
            left_seg1, left_len1 = self.checksegment(
                (segment[0], subgoal_left), obstacles, recursionlength + 1, target, field_bounds
            )
            left_seg2, left_len2 = self.checksegment(
                (subgoal_left, segment[1]), obstacles, recursionlength + 1, target, field_bounds
            )
            left_segments = left_seg1 + left_seg2
            left_length = left_len1 + left_len2

            # recursively check subsegments for right side
            right_seg1, right_len1 = self.checksegment(
                (segment[0], subgoal_right), obstacles, recursionlength + 1, target, field_bounds
            )
            right_seg2, right_len2 = self.checksegment(
                (subgoal_right, segment[1]), obstacles, recursionlength + 1, target, field_bounds
            )
            right_segments = right_seg1 + right_seg2
            right_length = right_len1 + right_len2
            # choose shorter path
            if not self.is_point_in_field(subgoal_left, field_bounds):
                return right_segments, right_length
            if not self.is_point_in_field(subgoal_right, field_bounds):
                return left_segments, left_length
            if right_length > left_length:
                return left_segments, left_length
            else:
                return right_segments, right_length

        else:
            # base case: no obstacle, return segment with its length
            segment_length = distance(segment[0], segment[1])
            return [segment], segment_length

    def _path_to(self, game: Game, robot_id: int, target: Tuple[float, float], field_bounds: FieldBounds):
        robot = game.friendly_robots[robot_id]
        our_pos = np.array([robot.p.x, robot.p.y])
        target = np.array(target)
        obstacles = self._get_obstacles(game, robot_id, our_pos, field_bounds)

        finaltrajectory = self.checksegment((our_pos, target), obstacles, 0, target, field_bounds)[0]
        return finaltrajectory
