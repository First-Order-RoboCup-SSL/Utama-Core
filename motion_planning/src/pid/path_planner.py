from typing import Tuple, Union, Optional, List

from entities.game import Game
from entities.game.game_object import Colour, GameObject, Robot as GameRobot
from entities.game.robot import Robot
from math import sin, cos, pi
import time
from itertools import product 
from math import dist
import random
from shapely import Point, LineString

from robot_control.src.find_best_shot import ROBOT_RADIUS 
class DynamicWindowPlanner:
    SIMULATED_TIMESTEP = 0.2 # seconds
    MAX_ACCELERATION = 2 # Measured in ms^2
    DIRECTIONS = [i*2*pi/16 for i in range(16)]
    OBSTACLE_RADIUS = 0.1 # in metres
    SAFE_OBSTACLES_RADIUS = 0.4
    ROBOT_RADIUS = 0.1


    def point_to_tuple(self, point: Point) -> tuple:
        """
        Convert a Shapely 2D Point into a tuple of (x, y).
        
        Args:
            point (Point): A Shapely Point object.
        
        Returns:
            tuple: A tuple (x, y) representing the coordinates of the point.
        """
        return (point.x, point.y)

    def __init__(self, game: Game):
        self._game = game
        self._friendly_colour = Colour.YELLOW if game.my_team_is_yellow else Colour.Blue
        self.target = None
        self.waypoints = []
        self.par = dict()

    def _closest_obstacle(self, robot_id: int, pos: Union[Point,LineString]) -> float:
        closest = float("inf")
        for r in self._get_obstacles(robot_id):
            rpos = Point(r.x, r.y)
            closest = min(closest, rpos.distance(pos))

        return closest
    
    def _adjust_segment_for_robot_radius(self, seg: LineString) -> LineString:
        current_robot_seg_interect = seg.interpolate(self.ROBOT_RADIUS)
        return LineString([current_robot_seg_interect, seg.interpolate(1, normalized=True)])
             

    def rrt_path_to(self, friendly_robot_id: int, target: Tuple[float, float], max_iterations: int = 10000) -> Optional[List[Point]]:
        """
        Generate a path to the target using the Rapidly-exploring Random Tree (RRT) algorithm.
        
        Args:
            friendly_robot_id (int): The ID of the friendly robot.
            target (Tuple[float, float]): The target coordinates (x, y).
        
        Returns:
            list: A list of waypoints (x, y) from the start to the target.
        
        The RRT algorithm works by randomly sampling points in the space and connecting them to the nearest existing point in the tree, 
        ensuring that the path does not collide with any obstacles. If a path is found that reaches close enough to the target, 
        the function returns the waypoints of the path.
        """
        # Initial robot position
        robot = self._game.friendly_robots[friendly_robot_id]
        start = Point(robot.x, robot.y)
        goal = Point(target[0], target[1])
        ROBOT_DIAMETER = 0.2

        # Check if path is even possible, if target is inside, just do start
        if self._closest_obstacle(friendly_robot_id, goal) < ROBOT_DIAMETER / 2:
            print(f"Path to target {target} is impossible due to an obstacle being too close.")
            return [start]

        # Already there - no work to do
        if start.distance(goal) < ROBOT_DIAMETER / 2:
            return [goal]

        # See if we can straight line it
        direct_path = LineString([start, goal])
        adjusted_direct_path = self._adjust_segment_for_robot_radius(direct_path)
        if self._closest_obstacle(friendly_robot_id, adjusted_direct_path) > self.SAFE_OBSTACLES_RADIUS:
            return [goal]

        explore_bias = 0.3
        # Initialize the tree with the start point
        parent_map = {start: None}
        path_found = False

        for _ in range(max_iterations):
            if random.random() < explore_bias:
                random_point = Point(random.uniform(-4.4, 4.4), random.uniform(-2.15, 2.15))
            else:
                random_point = goal

            # Find the closest point in the tree
            closest_point = min(parent_map.keys(), key=lambda p: p.distance(random_point))
            new_segment = LineString([closest_point, random_point])
            random_point = new_segment.interpolate(0.15)
            new_segment = LineString([closest_point, random_point])
            adjusted_new_segment = self._adjust_segment_for_robot_radius(new_segment)

            if self._closest_obstacle(friendly_robot_id, adjusted_new_segment) < self.SAFE_OBSTACLES_RADIUS:
                continue

            # Check if the new point is close enough to the goal to stop
            if adjusted_new_segment.distance(goal) < 0.3:
                parent_map[random_point] = closest_point
                path_found = True
                break

            parent_map[random_point] = closest_point

        if path_found:
            self.par = parent_map
            nearest_goal = min(parent_map.keys(), key=lambda p: p.distance(goal))
            path = [nearest_goal]
            current_point = nearest_goal

            while current_point != start:
                path.append(parent_map[current_point])
                current_point = parent_map[current_point]

            path = path[::-1]
            compressed_path = [path[0], path[1]]

            while len(path) > 2:
                collision = False
                segment = LineString([compressed_path[-1], path[1]])
                closest_distance = self._closest_obstacle(friendly_robot_id, segment)

                if collision or closest_distance < self.SAFE_OBSTACLES_RADIUS or segment.length > 4:
                    compressed_path.append(path[0])

                path.pop(0)

            compressed_path.extend(path)
            return compressed_path

        else:
            # No good enough path found
            return None

    def path_to(self, friendly_robot_id: int, target: Tuple[float, float]) -> Tuple[float, float]:
        """
        Plan a path to the target for the specified friendly robot.
        
        Args:
            friendly_robot_id (int): The ID of the friendly robot.
            target (Tuple[float, float]): The target coordinates (x, y).
            pid_oren: PID controller for orientation.
            pid_trans: PID controller for translation.
        
        Returns:
            Tuple[float, float]: The next waypoint coordinates (x, y) or the target if already reached.
        """

        robot: Robot = self._game.friendly_robots[friendly_robot_id]

        start_x, start_y = robot.x, robot.y
        if dist((start_x, start_y), target) <= self.SAFE_OBSTACLES_RADIUS:
            
            return target
        if target == self.target:
            if self.waypoints is None:
                print("NO PATH FOUND!!!!")
                return start_x, start_y
            elif not self.waypoints:
                return target
            
            if dist((start_x, start_y), self.waypoints[0]) <= 0.4:
                return self.waypoints.pop(0)
            else:
                return self.waypoints[0]
        else:
            self.target = target

            rrt_path = self.rrt_path_to(friendly_robot_id, target)
            if rrt_path is None:
                print("NO PATH FOUND!!!!")
                return start_x, start_y
            self.waypoints = list(map(self.point_to_tuple, rrt_path))
            
            if dist((start_x, start_y), self.waypoints[0]) <= 0.4:
                return self.waypoints.pop(0)
            else:
                return self.waypoints[0]


    
    def local_planning(self, friendly_robot_id: int, target: Tuple[float, float]):
        velocity = self._game.get_object_velocity(GameRobot(self._friendly_colour, friendly_robot_id))
        # DEPRECATED
        if velocity is None:
            # If no data, assume it is still :) 
            velocity = 0,0
        
        # Calculate the allowed velocities in this frame, 
        delta_vel = DynamicWindowPlanner.SIMULATED_TIMESTEP * DynamicWindowPlanner.MAX_ACCELERATION
        best_score = float("-inf")
        best_move = start_x, start_y
        # print("REACHED", start_x, start_y, velocity, delta_vel)

        for ang, vel_scale in product(DynamicWindowPlanner.DIRECTIONS, [1] ): #i/10 for i in range(1,11)
            ss = time.time()
            segment = self._get_motion_segment((start_x, start_y), velocity, delta_vel*vel_scale, ang)
            # Evaluate this segment, avoiding obstalces 
            score = self._evaluate_segment(friendly_robot_id, segment, Point(target[0], target[1]))
            # print("TARGET", segment, score)

            if score > best_score:
                best_score = score
                best_move = segment.coords[1]
        return best_move

    def _get_obstacles(self, robot_id):
        return self._game.friendly_robots[:robot_id] + self._game.friendly_robots[robot_id+1:] + self._game.enemy_robots

    def _evaluate_segment(self, robot_id: int, segment: LineString, target: Point) -> float:
        # Direct distance to target first, then we add the obstacles 
        # print(segment, target)
        score = target.distance(Point(segment.coords[0])) - target.distance(Point(segment.coords[1]))
        # print("BASE SCORE", score)
        # Need to calculate whether we will pass through any obstacles, check the closest point 

        for r in self._get_obstacles(robot_id):
            # See how close we pass to this robot
            rpos = Point(r.x, r.y)
            closest_distance = rpos.distance(segment)
            # print(closest_distance)

            if closest_distance < self.OBSTACLE_RADIUS:
                score = float("-inf")
            elif closest_distance < self.SAFE_OBSTACLES_RADIUS:
                score -= closest_distance - (self.SAFE_OBSTACLES_RADIUS - self.OBSTACLE_RADIUS)
 
        return score
                
    def _get_motion_segment(self, rpos: Tuple[float, float], rvel: Tuple[float, float], delta_vel: float, ang: float) -> LineString:
        adj_vel_y = rvel[1]*DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel*sin(ang)
        adj_vel_x = rvel[0]*DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel*cos(ang)
        end_y = adj_vel_y + rpos[1]
        end_x = adj_vel_x + rpos[0]

        return LineString([(rpos[0], rpos[1]), (end_x, end_y)])


