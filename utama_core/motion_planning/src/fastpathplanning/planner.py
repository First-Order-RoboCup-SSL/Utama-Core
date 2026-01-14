import math
from typing import List, Tuple

import numpy as np  # type: ignore

from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import distance, rotate_vector
from utama_core.motion_planning.src.fastpathplanning.config import (
    fastpathplanningconfig as config,
)
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Compute the shortest distance between a point and a line segment."""
    seg_vec = seg_end - seg_start
    t = np.clip(np.dot(point - seg_start, seg_vec) / np.dot(seg_vec, seg_vec), 0, 1)
    proj = seg_start + t * seg_vec
    return np.linalg.norm(point - proj)


class FastPathPlanner:
    def __init__(self, env: SSLStandardEnv):
        self._env = env
        self.config = config
        self.OBSTACLE_CLEARANCE = self.config.OBSTACLE_CLEARANCE
        self.LOOK_AHEAD_RANGE = self.config.LOOK_AHEAD_RANGE
        self.SUBGOAL_DISTANCE = self.config.SUBGOAL_DISTANCE

    def _get_obstacles(self, game: Game, robot_id: int, robotpos, target) -> List[np.ndarray]:
        friendly_obstacles = [robot for robot in game.friendly_robots.values() if robot.id != robot_id]
        robots = friendly_obstacles + list(game.enemy_robots.values())
        obstaclelist = []
        for r in robots:
            if r.v.x != 0.0 and r.v.y != 0.0:
                for i in range(0, 2):
                    velocity = np.array([r.v.x, r.v.y])
                    unitvec = velocity / np.linalg.norm(velocity)
                    point = np.array([r.p.x, r.p.y]) + i * unitvec * self.OBSTACLE_CLEARANCE
                    if (
                        distance(point, robotpos) < self.LOOK_AHEAD_RANGE
                        and distance(point, robotpos) > self.OBSTACLE_CLEARANCE
                        and distance(point, target) > self.OBSTACLE_CLEARANCE
                    ):
                        self._env.draw_point(point[0], point[1], width=20)
                        obstaclelist.append(point)
        return obstaclelist

    def _find_subgoal(self, robotpos, target, obstaclepos, obstacles, recursionfactor, multiple) -> np.array:
        direction = target - robotpos

        if recursionfactor % 2 == 1:
            perp_dir = rotate_vector(direction[0], direction[1], math.pi / 2)
        else:
            perp_dir = rotate_vector(direction[0], direction[1], math.pi * 3 / 2)

        unitvec = perp_dir / np.linalg.norm(perp_dir)
        subgoal = obstaclepos + self.SUBGOAL_DISTANCE * unitvec * multiple
        for o in obstacles:
            if distance(o, subgoal) < self.OBSTACLE_CLEARANCE:
                subgoal = self._find_subgoal(
                    robotpos, target, obstaclepos, obstacles, recursionfactor + 1, multiple + 1
                )
        return subgoal

    def collides(
        self, segment: Tuple, obstacles, target
    ):  # returns None if no obstacles, else it returns the closest obstacle.
        closestobstacle = None
        tempdistance = distance(segment[0], segment[1])
        for o in obstacles:
            if point_to_segment_distance(o, segment[0], segment[1]) < self.OBSTACLE_CLEARANCE:
                if (closestobstacle is None or distance(o, segment[0]) < tempdistance) and distance(
                    o, segment[1]
                ) > self.OBSTACLE_CLEARANCE:
                    tempdistance = distance(segment[0], o)
                    closestobstacle = o
        print(closestobstacle)
        return closestobstacle

    def _trajectory_length(self, trajectory):
        trajectory_legnth = 0
        for i in trajectory:
            trajectory_legnth += distance(i[0], i[1])
        return trajectory_legnth

    def checksegment(
        self, segment: Tuple, obstacles, recursionlength, target
    ):  # if there are obstacles in the segment, it divdes, the segment into two segments(initial_pos, subgoal) and (subgoal, target_pos), else returns the original segment.
        closestobstacle = self.collides(segment, obstacles, target)
        if closestobstacle is not None and recursionlength < 5:
            subgoal = []
            subgoal.append(self._find_subgoal(segment[0], segment[1], closestobstacle, obstacles, 1, 1))
            subgoal.append(self._find_subgoal(segment[0], segment[1], closestobstacle, obstacles, 0, 1))
            subseg_a1 = self.checksegment((segment[0], subgoal[0]), obstacles, recursionlength + 1, target)
            subseg_a2 = self.checksegment((subgoal[0], segment[1]), obstacles, recursionlength + 1, target)
            subseg_b1 = self.checksegment((segment[0], subgoal[1]), obstacles, recursionlength + 1, target)
            subseg_b2 = self.checksegment((subgoal[1], segment[1]), obstacles, recursionlength + 1, target)
            joined_sega = subseg_a1 + subseg_a2
            joined_segb = subseg_b1 + subseg_b2
            if self._trajectory_length(joined_sega) <= self._trajectory_length(joined_segb):
                return joined_sega
            else:
                return joined_segb
        else:
            return [segment]

    def _path_to(self, game: Game, robot_id: int, target: Tuple[float, float]):
        robot = game.friendly_robots[robot_id]
        our_pos = np.array([robot.p.x, robot.p.y])
        target = np.array(target)
        obstacles = self._get_obstacles(game, robot_id, our_pos, target)
        finaltrajectory = self.checksegment((our_pos, target), obstacles, 0, target)
        # for i in finaltrajectory:
        #      self._env.draw_point(i[1][0], i[1][1], width=10)
        return finaltrajectory


# Here finaltrajectory is the final calculated trajectory which is a list consisting of different segements of the trajectory. Each segment is represented using a tuple
