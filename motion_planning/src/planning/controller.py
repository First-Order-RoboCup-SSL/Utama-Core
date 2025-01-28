from typing import Tuple

from numpy import full
from entities.game.game import Game
from entities.game.robot import Robot
from motion_planning.src.planning.path_planner import RRTPlanner, DynamicWindowPlanner, target_inside_robot_radius
from math import dist
import time
class HybridWaypointMotionController:
    """Motion controller that takes two path planners, one which produces locally good moves
       and another which produces a list of waypoints to the target the aim of the class
       is to track state such as switching waypoints. 
       """
    DEFAULT_TAKEN_WAYPOINTS = 30
    def __init__(self, num_robots: int, game: Game, env):
        self.num_robots = num_robots
        self._waypoint_producer = RRTPlanner(game)
        self._game = game
        self._local_planner = DynamicWindowPlanner(game)
        self._real_targets = [None for _ in range(num_robots)]
        self._intermediate_targets = [[] for _ in range(num_robots)]
        self._num_waypoints_to_take = [HybridWaypointMotionController.DEFAULT_TAKEN_WAYPOINTS for _ in range(num_robots)]
        self._num_waypoints_taken = [0 for _ in range(num_robots)]

        # DEBUG ONLY
        self._env = env

    def path_to(self, target: Tuple[float, float], robot_id: int) -> Tuple[int, int]:
        robot: Robot = self._game.friendly_robots[robot_id]

        start_position = robot.x, robot.y
        if target_inside_robot_radius(start_position, target):
            self._intermediate_targets[robot_id] = []
            self._num_waypoints_to_take[robot_id] = HybridWaypointMotionController.DEFAULT_TAKEN_WAYPOINTS
            self._num_waypoints_taken[robot_id] = 0
            print("REACHED TARGET")
            assert target is not None
            return target

        if target != self._real_targets[robot_id]:
            self._real_targets[robot_id] = target
            self._intermediate_targets[robot_id] = []
            self._num_waypoints_to_take[robot_id] = HybridWaypointMotionController.DEFAULT_TAKEN_WAYPOINTS
            self._num_waypoints_taken[robot_id] = 0
            print("TARGET CHANGED")

        if not self._intermediate_targets[robot_id] or self._num_waypoints_taken[robot_id] >= self._num_waypoints_to_take[robot_id]:
            best_move, best_score = self._local_planner.path_to(robot_id, target)
            print("REACHED")
            if best_score >= 0.2:
                print("GOOD DWA", best_move, best_score)
                assert best_move is not None
                return best_move
            else:
                # We are stuck, calculate a path using RRT
                print("STUCK, runnning RRT", best_move)
                if not self._intermediate_targets[robot_id]:
                    start = time.time()
                    self._intermediate_targets[robot_id] = self._waypoint_producer.path_to(robot_id, target)
                    print("TOOK:", time.time)
                print(self._intermediate_targets[robot_id])
                # Double when we got stuck
                self._num_waypoints_to_take[robot_id] *= 2
                self._num_waypoints_taken[robot_id] = 0

        if self._intermediate_targets[robot_id]:
            # We were stuck and are heading to an intermediate target calculated by RRT
            # Use DWA for this too

            next_stop = self._intermediate_targets[robot_id][0]
            if dist(start_position, next_stop) < 0.2:
                self._intermediate_targets[robot_id].pop(0)
                self._num_waypoints_taken[robot_id] += 1
                next_stop = self._intermediate_targets[robot_id][0] if self._intermediate_targets[robot_id] else target
            assert next_stop is not None
            return self._local_planner.path_to(robot_id, next_stop)[0]
        else:
            print("You are in deep trouble lad")
            return target



