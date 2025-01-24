from typing import Tuple, Union, Optional, List

from entities.game import Game
from entities.game.game_object import Colour, GameObject, Robot as GameRobot
from entities.game.robot import Robot
from math import sin, cos, pi, sqrt
import time
from itertools import product 
from math import dist, exp
import random
from shapely import Point, LineString

from robot_control.src.find_best_shot import ROBOT_RADIUS 

"""
TODO - RRT is too slow and not adaptive enough,
use RTT waypoints as part of the Dynamic window Approach heuristic
This should give enough adaptivity 
to avoid obstacles whilst being a globally decent path

Edge cases:
    target inside obstacle
    target too close to obstacle (within// Install Vtune
Drift
Cleanup so that it takes a robot for the path every time
Magic numbers

Motion controller stateful with waypoints - take robot index or separate one for each robot
Inner stateless one does the path planning
 -> Take local planning if possible
 -> Otherwise make a new global tree and go towards next waypoint until you get there and then use local planning
Fix clearance
Make Field stuff static and fix the dimensions (2.15)
Test with motion
Slow motion and never gets there.
"""

N_DIRECTIONS = 16
class DynamicWindowPlanner:
    SIMULATED_TIMESTEP = 0.2 # seconds
    MAX_ACCELERATION = 2 # Measured in ms^2
    DIRECTIONS = [i * 2 * pi / N_DIRECTIONS for i in range(N_DIRECTIONS)]
    OBSTACLE_RADIUS = 0.1 # in metres
    SAFE_OBSTACLES_RADIUS = 0.25
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
            random_point = new_segment.interpolate(0.1)
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

        if dist((start_x, start_y), target) < 1.5*ROBOT_RADIUS:
            return target
        
        return self.local_planning(friendly_robot_id, target)


    
    def local_planning(self, friendly_robot_id: int, target: Tuple[float, float]):
        velocity = self._game.get_object_velocity(GameRobot(self._friendly_colour, friendly_robot_id))

        if velocity is None:
            # If no data, assume it is still :) 
            velocity = 0,0
        # print(velocity, "HERE")
        # Calculate the allowed velocities in this frame
        delta_vel = DynamicWindowPlanner.SIMULATED_TIMESTEP * DynamicWindowPlanner.MAX_ACCELERATION
        best_score = float("-inf")
        robot: Robot = self._game.friendly_robots[friendly_robot_id]

        start_x, start_y = robot.x, robot.y
        best_move = start_x, start_y
        # print()
        # print("---------------------------------")
        # print()
        sf = 1

        while best_score < 0 and sf > 0.001:
            for ang, vel_scale in product(DynamicWindowPlanner.DIRECTIONS, [sf]): #i/10 for i in range(1,11)
                segment = self._get_motion_segment((start_x, start_y), velocity, delta_vel*vel_scale, ang)
                # Evaluate this segment, avoiding obstalces 
                score = self._evaluate_segment(friendly_robot_id, segment, Point(target[0], target[1]))
                                            # self._adjust_segment_for_robot_radius(segment), Point(target[0], target[1]))
                # print(segment, score)
                # print()
                if score > best_score:
                    best_score = score
                    best_move = segment.coords[1]
            
            sf /= 2
            # print(best_score, sf)
        # print("MOVING", best_move)
        return best_move

    def _get_obstacles(self, robot_id):
        return self._game.friendly_robots[:robot_id] + self._game.friendly_robots[robot_id+1:] + self._game.enemy_robots

    def make_inf_long(self, segment: LineString):
        norm = segment.length
        endX, endY = segment.coords[1]
        startX, startY = segment.coords[0]
        new = Point(startX + (endX - startX) / norm * 18, startY + (endY - startY) / norm * 18)
        return LineString([segment.coords[0], new]) 

    def obstacle_penalty_function(self, x):
        # return (0.035 / (x - 0.18)) + 0.03/x
        return exp(-8 * (x - 0.22))    

    def target_closeness_function(self, x):
        return 4 * exp(-8 * x)

    def _evaluate_segment(self, robot_id: int, segment: LineString, target: Point) -> float:
        """Evaluate line segment; bigger score is better"""
        target_factor = target.distance(Point(segment.coords[0])) - target.distance(Point(segment.coords[1]))
        our_velocity_vector = (segment.coords[1][0] - segment.coords[0][0]) / self.SIMULATED_TIMESTEP, (segment.coords[1][1] - segment.coords[0][1]) / self.SIMULATED_TIMESTEP
        if our_velocity_vector is None:
            our_velocity_vector = (0, 0)
        
        our_position = self._game.get_robot_pos(self._friendly_colour == Colour.YELLOW, robot_id)

        obstacle_factor = 0

        for r in self._get_obstacles(robot_id):            
            their_velocity_vector = self._game.get_object_velocity(GameRobot(self._friendly_colour if r.is_friendly else Colour.invert(self._friendly_colour), r.id))
            # print(our_velocity_vector, their_velocity_vector)

            if their_velocity_vector is None:
                their_velocity_vector = (0, 0)

            their_position = (r.x, r.y)
            # print("O/T pos", our_position, their_position)

            diff_v_x = our_velocity_vector[0] - their_velocity_vector[0]
            diff_p_x = our_position[0] - their_position[0]
            diff_v_y = our_velocity_vector[1] - their_velocity_vector[1]
            diff_p_y = our_position[1] - their_position[1]
            # print("POSN", their_position)
            # print(f"DIFF {diff_p_x} {diff_v_x} {diff_p_y} {diff_v_y}")

            if (denom := (diff_v_x * diff_v_x + diff_v_y * diff_v_y)) != 0:
                t = (-diff_v_x * diff_p_x - diff_v_y * diff_p_y) / denom
                # print("NUM", (-diff_v_x * diff_p_x - diff_v_y * diff_p_y))
                # print("DENOM", denom)
                # print("TIME", t)
                if t > 0:
                    d_sq = (diff_p_x + t * diff_v_x) ** 2 + (diff_p_y + t * diff_v_y) ** 2
                    # print("D_SQ", d_sq)
                    # print("OBS FACt", self.exp_decay(t) * self.exp_decay(d_sq))
                    obstacle_factor = max(obstacle_factor,  self.obstacle_penalty_function(d_sq)) # self.exp_decay(d_sq)) # self.exp_decay(t) * self.exp_decay(d_sq))

        score = 5 * target_factor - obstacle_factor + self.target_closeness_function(target.distance(segment)) # Point(segment.coords[1])# - 2 * target.distance(Point(segment.coords[1])) + 
        # print("FACT", target_factor, obstacle_factor, score, segment)
        return score

        # # Distance travelled towards target by this segment

        # # Need to calculate whether we will pass through any obstacles, check the closest point 
        #     # See how close we pass to this robot's predicted path
        #     predicted = None
        #     if r.is_friendly:
        #         predicted = self._game.predict_object_pos_after(self.SIMULATED_TIMESTEP, GameRobot(self._friendly_colour, r.id))
        #         velocity = self._game.get_object_velocity(GameRobot(self._friendly_colour, r.id))

        #         print("VEL: ", velocity)
        #     # else:
        #     #     predicted = self._game.predict_object_pos_after(self.SIMULATED_TIMESTEP, GameRobot(Colour.BLUE, r.id))

        #     if predicted is None:
        #         rpos = Point(r.x, r.y)
        #         closest_distance = rpos.distance(segment)
        #     else:
        #         rpos = Point(r.x, r.y)
        #         rpred_pos = Point([predicted[0], predicted[1]])
        #         closest_distance = rpred_pos.distance(segment)
        #         print("PRED", predicted, rpred_pos, segment)
        #     # print("distance")
        #     # print(rpos)
        #     # print(closest_distance)

        #     if closest_distance < self.OBSTACLE_RADIUS:
        #         score = float("-inf")
        #     obst_score += exp(-2*(closest_distance-0.07))
        # print("OBSTACLE SCORE: ", obst_score)
        # return score
                
    def _get_motion_segment(self, rpos: Tuple[float, float], rvel: Tuple[float, float], delta_vel: float, ang: float) -> LineString:
        adj_vel_y = rvel[1]*DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel*sin(ang)
        adj_vel_x = rvel[0]*DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel*cos(ang)
        end_y = adj_vel_y + rpos[1]
        end_x = adj_vel_x + rpos[0]

        return LineString([(rpos[0], rpos[1]), (end_x, end_y)])


