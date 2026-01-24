import math
from typing import Tuple

import numpy as np  # type: ignore

from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import (
    distance,
    distance_between_line_segments,
    distance_point_to_segment,
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

    def _get_obstacles(self, game: Game, robot_id: int, our_pos, target):
        friendly_obstacles = [robot for robot in game.friendly_robots.values() if robot.id != robot_id]
        robots = friendly_obstacles + list(game.enemy_robots.values())
        obstaclelist = []
        for r in robots:
            robot_pos = np.array([r.p.x, r.p.y])
            if (
                distance(our_pos, robot_pos) < self.LOOK_AHEAD_RANGE
                and distance(robot_pos, our_pos) > self.OBSTACLE_CLEARANCE
                and distance(robot_pos, target) > self.OBSTACLE_CLEARANCE
            ):
                if abs(r.v.x) > 10e-10 and abs(r.v.y) > 10e-10:
                    velocity = np.array([r.v.x, r.v.y])
                    point = np.array([r.p.x, r.p.y]) + velocity * self.PROJECTEDFRAMES / 60
                    obstalcesegment = (robot_pos, point)
                    obstaclelist.append(obstalcesegment)
                else:
                    obstaclelist.append((robot_pos, robot_pos))
        return obstaclelist

    def _find_subgoal(
        self,
        robot_pos,
        target,
        closest_obstacle,
        obstacles,
        recursionfactor,
        multiple,
    ) -> np.array:
        direction = target - robot_pos

        if recursionfactor % 2 == 1:
            perp_dir = rotate_vector(direction[0], direction[1], math.pi / 2)
        else:
            perp_dir = rotate_vector(direction[0], direction[1], math.pi * 3 / 2)

        unitvec = perp_dir / np.linalg.norm(perp_dir)
        obstaclepos = (closest_obstacle[0] + closest_obstacle[1]) / 2
        subgoal = obstaclepos + self.SUBGOAL_DISTANCE * unitvec * multiple

        for o in obstacles:
            if distance_point_to_segment(subgoal, o[0], o[1]) < self.OBSTACLE_CLEARANCE:
                subgoal = self._find_subgoal(
                    robot_pos,
                    target,
                    closest_obstacle,
                    obstacles,
                    recursionfactor + 1,
                    multiple + 1,
                )
        return subgoal

    def collides(
        self, segment: Tuple, obstacles, target
    ):  # returns None if no obstacles, else it returns the closest obstacle.
        closest_obstacle = None
        tempdistance = distance(segment[0], segment[1])
        for o in obstacles:
            # print('hello',o,segment)
            if distance_between_line_segments(o[0], o[1], segment[0], segment[1]) < self.OBSTACLE_CLEARANCE:
                obstacledistance = distance_between_line_segments(o[0], o[1], segment[0], segment[1])
                if closest_obstacle is None or obstacledistance < tempdistance:
                    tempdistance = obstacledistance
                    closest_obstacle = o

        return closest_obstacle

    def _trajectory_length(self, trajectory):
        trajectory_legnth = 0
        for i in trajectory:
            trajectory_legnth += distance(i[0], i[1])
        return trajectory_legnth

    def checksegment(
        self, segment: Tuple, obstacles, recursionlength, target
    ):  # if there are obstacles in the segment, it divdes, the segment into two segments(initial_pos, subgoal) and (subgoal, target_pos), else returns the original segment.
        closest_obstacle = self.collides(segment, obstacles, target)
        if closest_obstacle is not None and recursionlength < self.MAXRECURSIONLENGTH:
            subgoal = []
            subgoal.append(self._find_subgoal(segment[0], segment[1], closest_obstacle, obstacles, 1, 1))
            subgoal.append(self._find_subgoal(segment[0], segment[1], closest_obstacle, obstacles, 0, 1))
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
