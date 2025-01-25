from typing import Tuple, Union, Optional, List
from entities.game import Game
from entities.game.game_object import Colour, Robot as GameRobot
from entities.game.robot import Robot
from math import sin, cos, pi, dist, exp
import random
from shapely.geometry import Point, LineString
from robot_control.src.find_best_shot import ROBOT_RADIUS

ROBOT_DIAMETER = 0.2

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


def point_to_tuple(point: Point) -> tuple:
    """
    Convert a Shapely 2D Point into a tuple of (x, y).
    
    Args:
        point (Point): A Shapely Point object.
    
    Returns:
        tuple: A tuple (x, y) representing the coordinates of the point.
    """
    return (point.x, point.y)

def smooth_path(points: List[Tuple[float, float]], smoothing_factor: float = 0.1) -> List[Tuple[float, float]]:
    """
    Smooth a path using a simple moving average.
    
    Args:
        points (List[Tuple[float, float]]): A list of 2D points representing the path.
        smoothing_factor (float): A parameter to control the amount of smoothing.
        
    Returns:
        List[Tuple[float, float]]: A list of 2D points representing the smoothed path.
    """
    if len(points) < 3:
        return points

    smoothed_points = []
    window_size = max(2, int(smoothing_factor * len(points)))

    for i in range(len(points)):
        start = max(0, i - window_size)
        end = min(len(points), i + window_size)
        window = points[start:end]
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        smoothed_points.append((avg_x, avg_y))
    
    # Reduce points that are less than 1 unit away from each other
    # reduced_points = [smoothed_points[0]]
    # for point in smoothed_points[1:]:
    #     if dist(reduced_points[-1], point) >= 1:
    #         reduced_points.append(point)

    # # Further reduce points by checking line of sight
    # final_points = [reduced_points[0]]
    # for i in range(1, len(reduced_points)):
    #     if i == len(reduced_points) - 1 or not LineString([final_points[-1], reduced_points[i + 1]]).is_simple:
    #         final_points.append(reduced_points[i])

    return smoothed_points

class RRTPlanner:
    SAFE_OBSTACLES_RADIUS = 0.28
    STOPPING_DISTANCE = 0.2
    EXPLORE_BIAS = 0.1
    STEP_SIZE = 0.25
    GOOD_ENOUGH_REL = 1.2
    GOOD_ENOUGH_ABS = 1

    def __init__(self, game: Game):
        self._game = game
        self._friendly_colour = Colour.YELLOW if game.my_team_is_yellow else Colour.Blue
        self.target = None
        self.waypoints = []
        self.par = dict()

    def _get_obstacles(self, robot_id):
        return self._game.friendly_robots[:robot_id] + self._game.friendly_robots[robot_id+1:] + self._game.enemy_robots

    def _closest_obstacle(self, robot_id: int, pos: Union[Point, LineString]) -> float:
        closest = float("inf")
        for r in self._get_obstacles(robot_id):
            rpos = Point(r.x, r.y)
            closest = min(closest, rpos.distance(pos))
        return closest

    def _adjust_segment_for_robot_radius(self, seg: LineString) -> LineString:
        current_robot_seg_interect = seg.interpolate(ROBOT_RADIUS)
        return LineString([current_robot_seg_interect, seg.interpolate(1, normalized=True)])


    def _propagate(self, parent_map, cost_map, parent):
        for p in parent_map.keys():
            if parent_map[p] == parent:
                cost_map[p] = cost_map[parent] + p.distance(parent)
                self._propagate(parent_map, cost_map, p)

    def path_to(self, friendly_robot_id: int, target: Tuple[float, float], max_iterations: int = 1000) -> Optional[List[Tuple[float, float]]]:
        """
        Generate a path to the target using the Rapidly-exploring Random Tree (RRT) algorithm.
        
        Args:
            friendly_robot_id (int): The ID of the friendly robot.
            target (Tuple[float, float]): The target coordinates (x, y).
            max_iterations (int): Maximum number of iterations for the RRT algorithm.
        
        Returns:
            Optional[List[Tuple[float, float]]]: A list of waypoints (x, y) from the start to the target.
        """
        robot = self._game.friendly_robots[friendly_robot_id]
        start = Point(robot.x, robot.y)
        goal = Point(target[0], target[1])

        if self._closest_obstacle(friendly_robot_id, goal) < ROBOT_DIAMETER / 2:
            return [(start.x, start.y)]

        if start.distance(goal) < ROBOT_DIAMETER / 2:
            return [(goal.x, goal.y)]

        direct_path = LineString([start, goal])
        adjusted_direct_path = self._adjust_segment_for_robot_radius(direct_path)
        if self._closest_obstacle(friendly_robot_id, adjusted_direct_path) > 3*self.SAFE_OBSTACLES_RADIUS:
            return [(goal.x, goal.y)]

        parent_map = {start: None}
        cost_map = {start: 0}
        path_found = False

        for _ in range(max_iterations):
            if random.random() < self.EXPLORE_BIAS:
                rand_point = Point(random.uniform(-4.4, 4.4), random.uniform(-3, 3))
            else:
                rand_point = Point(target[0], target[1])
            

            closest_point = min(parent_map.keys(), key=lambda p: p.distance(rand_point))
            new_segment = LineString([closest_point, rand_point])
            rand_point = new_segment.interpolate(self.STEP_SIZE)
            new_segment = LineString([closest_point, rand_point])

            if self._closest_obstacle(friendly_robot_id, new_segment) > self.SAFE_OBSTACLES_RADIUS and rand_point not in parent_map:
                # Choose the best parent node 
                best_parent = closest_point
                min_cost = cost_map[closest_point] + closest_point.distance(rand_point)
                for p in parent_map.keys():
                    if p.distance(rand_point) < 2*self.STEP_SIZE and cost_map[p] + p.distance(rand_point) < min_cost and self._closest_obstacle(friendly_robot_id, LineString([p, rand_point])) > self.SAFE_OBSTACLES_RADIUS:
                        best_parent = p
                        min_cost = cost_map[p] + p.distance(rand_point)
                
                cost_map[rand_point] = min_cost
                parent_map[rand_point] = best_parent

                # Now need to rewire the tree based on this 
                
                for p in parent_map.keys():
                    if p.distance(rand_point) < 2:
                        # Might be able to find a better path through rand_point
                        if cost_map[rand_point] + rand_point.distance(p) < cost_map[p] and self._closest_obstacle(friendly_robot_id, LineString([p, rand_point])) > self.SAFE_OBSTACLES_RADIUS:
                            cost_map[p] = cost_map[rand_point] + rand_point.distance(p)
                            parent_map[p] = rand_point
                            self._propagate(parent_map, cost_map, p)
                
                if rand_point.distance(goal) < self.STOPPING_DISTANCE and cost_map[rand_point] + rand_point.distance(goal) < cost_map.get(goal, float("inf")):
                    parent_map[goal] = rand_point
                    cost_map[goal] = cost_map[rand_point] + rand_point.distance(goal)
                    path_found = True
                    if cost_map[goal] <= self.GOOD_ENOUGH_ABS*goal.distance(start) or cost_map[goal] - goal.distance(start) < self.GOOD_ENOUGH_ABS:
                        break
        if path_found:
            self.par = parent_map
            path = []
            current_point = goal
            visited = set()
            while current_point is not None:
                if current_point in visited:
                    break
                visited.add(current_point)
                path.append(current_point)
                current_point = parent_map[current_point]
            path.reverse()
            return [(p.x, p.y) for p in path]

        return None


N_DIRECTIONS = 16
class DynamicWindowPlanner:
    SIMULATED_TIMESTEP = 0.2 # seconds
    MAX_ACCELERATION = 2 # Measured in ms^2
    DIRECTIONS = [i * 2 * pi / N_DIRECTIONS for i in range(N_DIRECTIONS)]
    ROBOT_RADIUS = 0.1

    def __init__(self, game: Game):
        self._game = game
        self._friendly_colour = Colour.YELLOW if game.my_team_is_yellow else Colour.Blue    

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

        # Calculate the allowed velocities in this frame
        delta_vel = DynamicWindowPlanner.SIMULATED_TIMESTEP * DynamicWindowPlanner.MAX_ACCELERATION
        best_score = float("-inf")
        robot: Robot = self._game.friendly_robots[friendly_robot_id]

        start_x, start_y = robot.x, robot.y
        best_move = start_x, start_y

        # sf is the scale factor for the velocity - we start with full velocity to prioritise speed
        # and then reduce it if we can't find a good segment, this allows the robot to dynamically adjust
        # its speed and path length to avoid obstacles
        sf = 1
        while best_score < 0 and sf > 0.05:
            for ang in DynamicWindowPlanner.DIRECTIONS: 
                segment = self._get_motion_segment((start_x, start_y), velocity, delta_vel*sf, ang)
                # Evaluate this segment, avoiding obstacles
                score = self._evaluate_segment(friendly_robot_id, segment, Point(target[0], target[1]))

                if score > best_score:
                    best_score = score
                    best_move = segment.coords[1]
            
            sf /= 2

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
        return exp(-8 * (x - 0.22))    

    def target_closeness_function(self, x):
        return 4 * exp(-8 * x)

    def _evaluate_segment(self, robot_id: int, segment: LineString, target: Point) -> float:
        """Evaluate line segment; bigger score is better"""
        # Distance travelled towards the target should be rewarded
        target_factor = target.distance(Point(segment.coords[0])) - target.distance(Point(segment.coords[1]))
        our_velocity_vector = (segment.coords[1][0] - segment.coords[0][0]) / self.SIMULATED_TIMESTEP, (segment.coords[1][1] - segment.coords[0][1]) / self.SIMULATED_TIMESTEP
        if our_velocity_vector is None:
            our_velocity_vector = (0, 0)
        
        our_position = self._game.get_robot_pos(self._friendly_colour == Colour.YELLOW, robot_id)

        # If we are too close to an obstacle, we should be penalised
        obstacle_factor = 0

        # Factor in obstacle velocity, do some maths to find their closest approach to us
        # and the time at which that happens. 
        for r in self._get_obstacles(robot_id):            
            their_velocity_vector = self._game.get_object_velocity(GameRobot(self._friendly_colour if r.is_friendly else Colour.invert(self._friendly_colour), r.id))

            if their_velocity_vector is None:
                their_velocity_vector = (0, 0)

            their_position = (r.x, r.y)

            diff_v_x = our_velocity_vector[0] - their_velocity_vector[0]
            diff_p_x = our_position[0] - their_position[0]
            diff_v_y = our_velocity_vector[1] - their_velocity_vector[1]
            diff_p_y = our_position[1] - their_position[1]

            if (denom := (diff_v_x * diff_v_x + diff_v_y * diff_v_y)) != 0:
                t = (-diff_v_x * diff_p_x - diff_v_y * diff_p_y) / denom
                if t > 0:
                    d_sq = (diff_p_x + t * diff_v_x) ** 2 + (diff_p_y + t * diff_v_y) ** 2

                    obstacle_factor = max(obstacle_factor,  self.obstacle_penalty_function(d_sq)*self.obstacle_penalty_function(t))

        # Adjust weights for the final score - this is done by tuning
        score = 5 * target_factor - obstacle_factor + self.target_closeness_function(target.distance(segment))
        return score

    def _get_motion_segment(self, rpos: Tuple[float, float], rvel: Tuple[float, float], delta_vel: float, ang: float) -> LineString:
        adj_vel_y = rvel[1]*DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel*sin(ang)
        adj_vel_x = rvel[0]*DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel*cos(ang)
        end_y = adj_vel_y + rpos[1]
        end_x = adj_vel_x + rpos[0]
        return LineString([(rpos[0], rpos[1]), (end_x, end_y)])


