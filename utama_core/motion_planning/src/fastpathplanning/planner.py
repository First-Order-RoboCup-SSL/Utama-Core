from typing import List, Tuple

import numpy as np  # type: ignore

from utama_core.entities.game import Game
from utama_core.motion_planning.src.fastpathplanning.config import (
    fastpathplanningconfig as config,
)
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv


def distance(a, b) -> float:
    return np.linalg.norm(a - b)


def rotate_vector(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return vec @ rot.T


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
        self.OBSTACLE_CLEARANCE = self.config.ROBOT_DIAMETER

    def _get_obstacles(self, game: Game, robot_id: int) -> List[np.ndarray]:
        friendly_obstacles = [robot for robot in game.friendly_robots.values() if robot.id != robot_id]
        robots = friendly_obstacles + list(game.enemy_robots.values())
        return [np.array([r.p.x, r.p.y]) for r in robots]

    def _find_subgoal(self, robotpos, target, obstaclepos, obstacles) -> np.array:
        direction = target - robotpos
        perp_dir = rotate_vector(direction, 90)
        unitvec = perp_dir / np.linalg.norm(perp_dir)
        subgoal = obstaclepos + self.OBSTACLE_CLEARANCE * unitvec * 3
        for o in obstacles:
            if distance(o, subgoal) < self.OBSTACLE_CLEARANCE:
                subgoal = obstaclepos - self.OBSTACLE_CLEARANCE * unitvec * 3

        return subgoal

    def collides(
        self, segment: Tuple, obstacles
    ):  # returns None if no obstacles, else it returns the closest obstacle.
        closestobstacle = None
        tempdistance = distance(segment[0], segment[1])
        for o in obstacles:
            if point_to_segment_distance(o, segment[0], segment[1]) < self.OBSTACLE_CLEARANCE:
                # print('obstavle in path')
                if closestobstacle is None or distance(0, segment[0]) < tempdistance:
                    tempdistance = distance(segment[0], o)
                    closestobstacle = o
        # print(segment[0], closestobstacle, segment[1])
        return closestobstacle

    def checksegment(
        self, segment: Tuple, obstacles
    ):  # if there are obstacles in the segment, it divdes, the segment into two segments(initial_pos, subgoal) and (subgoal, target_pos), else returns the original segment.
        closestobstacle = self.collides(segment, obstacles)
        if closestobstacle is not None:
            subgoal = self._find_subgoal(segment[0], segment[1], closestobstacle, obstacles)
            subseg_1 = self.checksegment((segment[0], subgoal), obstacles)
            subseg_2 = [(subgoal, segment[1])]
            joined_seg = subseg_1 + subseg_2
            return joined_seg
        else:
            return [segment]

    def _path_to(self, game: Game, robot_id: int, target: Tuple[float, float]):
        robot = game.friendly_robots[robot_id]
        our_pos = np.array([robot.p.x, robot.p.y])
        target = np.array(target)

        obstacles = self._get_obstacles(game, robot_id)
        finaltrajectory = self.checksegment((our_pos, target), obstacles)
        return finaltrajectory


# Here finaltrajectory is the final calculated trajectory which is a list consisting of different segements of the trajectory. Each segment is represented using a tuple
