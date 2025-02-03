from typing import Tuple, Union, Optional, List, Generator

from shapely import Polygon
from entities.game import Game
from entities.game.field import Field
from entities.game.game_object import Colour, Robot as GameRobot
from entities.game.robot import Robot
from math import sin, cos, pi, dist, exp
import random
from shapely.geometry import Point, LineString
from team_controller.src.config.settings import ROBOT_RADIUS
from shapely.affinity import rotate

import logging

logger = logging.getLogger(__name__)


ROBOT_DIAMETER = 2 * ROBOT_RADIUS

"""
TODO -
Edge cases:
    target inside obstacle
    target starts within / too close to obstacle 

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


def target_inside_robot_radius(
    rpos: Tuple[float, float], target: Tuple[float, float]
) -> bool:
    return dist(rpos, target) <= ROBOT_RADIUS


class RRTPlanner:
    """This class is a stateless planning class and should not be used on its own
    see the controllers class which provide state tracking and waypoint switching for these classes such as TimedSwitchController
    """
    # TODO - make these parameters configurable at runtime
    # TODO - Add support for avoiding goal areas - should be easy to use the Field object for this

    SAFE_OBSTACLES_RADIUS = (
        2 * ROBOT_RADIUS + 0.08
    )  # 2*ROBOT_RADIUS + 0.08 for wiggle room
    STOPPING_DISTANCE = 0.2  # When are we close enough to the goal to stop
    EXPLORE_BIAS = 0.1  # How often the tree does a random exploration
    STEP_SIZE = 0.15
    # acceptable relative and absolute error from the target (euclidian distance)
    GOOD_ENOUGH_REL = 1.2
    GOOD_ENOUGH_ABS = 1

    def __init__(self, game: Game):
        self._game = game
        self._friendly_colour = Colour.YELLOW if game.my_team_is_yellow else Colour.Blue
        self.waypoints = []
        self.par = dict()

    def _reduce_waypoints(
        self, waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        # Takes a list of waypoints and removes the middle waypoint in a set of three if they are nearly collinear
        # It does this by checking the distance of the middle waypoint from the line formed by the other two
        # This is to reduce the number of waypoints and make the path smoother
        if len(waypoints) < 3:
            return waypoints
        new_waypoints = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            skipped_segment = LineString([new_waypoints[-1], waypoints[i + 1]])
            if (
                skipped_segment.distance(Point(waypoints[i])) > 0.03
                or skipped_segment.length > 2
            ):
                new_waypoints.append(waypoints[i])
        new_waypoints.append(waypoints[-1])
        return new_waypoints

    def _get_obstacles(self, robot_id):
        return (
            self._game.friendly_robots[:robot_id]
            + self._game.friendly_robots[robot_id + 1 :]
            + self._game.enemy_robots
        )

    def _closest_obstacle(self, robot_id: int, pos: Union[Point, LineString]) -> float:
        closest = float("inf")
        for r in self._get_obstacles(robot_id):
            rpos = Point(r.x, r.y)
            closest = min(closest, rpos.distance(pos))
        return closest

    def _adjust_segment_for_robot_radius(self, seg: LineString) -> LineString:
        current_robot_seg_interect = seg.interpolate(ROBOT_RADIUS)
        return LineString(
            [current_robot_seg_interect, seg.interpolate(1, normalized=True)]
        )

    def _propagate(self, parent_map, cost_map, parent):
        # Technically should push costs down to the children when we find a better path to the parent
        # In practice this is just really slow and doesn't make a huge difference in our use case
        for p in parent_map.keys():
            if parent_map[p] == parent:
                cost_map[p] = cost_map[parent] + p.distance(parent)
                self._propagate(parent_map, cost_map, p)

    def _get_nearby(self, point: Point) -> List[Point]:
        return [p for p in self._get_adj_cells(point)]

    def _get_adj_cells(self, point: Point) -> Generator[Point, None, None]:
        gx, gy = self._compress_point(point)
        for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            if 0 <= gx + dx < 9 and 0 <= gy + dy < 6:
                for p in self.grid[gy + dy][gx + dx]:
                    yield p

    def _compress_point(self, point: Point) -> Tuple[int, int]:
        return int((point.x + Field.HALF_LENGTH) // 1), int(
            (point.y + Field.HALF_WIDTH) // 1
        )

    def _add_compressed_point(self, point: Point):
        cp = self._compress_point(point)
        self.grid[cp[1]][cp[0]].append(point)

    def path_to(
        self,
        friendly_robot_id: int,
        target: Tuple[float, float],
        max_iterations: int = 3000,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Generate a path to the target using the Rapidly-exploring Random Tree Star (RRT*) algorithm.

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
            logger.debug(
                "RRT Planner: Goal is inside obstacle radius of goal - no path there, return start"
            )

            return [(start.x, start.y)]

        if start.distance(goal) < ROBOT_DIAMETER / 2:
            logger.debug(
                "RRT Planner: Goal is inside robot radius of goal - already there, return goal"
            )
            return [(goal.x, goal.y)]

        direct_path = LineString([start, goal])
        adjusted_direct_path = self._adjust_segment_for_robot_radius(direct_path)

        # Need more than the safe obstacle radius as at high speeds this does not work
        if (
            self._closest_obstacle(friendly_robot_id, adjusted_direct_path)
            > 3 * self.SAFE_OBSTACLES_RADIUS
        ):
            logger.debug("RRT Planner: Goal direct line of sight - Go straight there")

            return [(goal.x, goal.y)]

        self.grid = [[[] for _ in range(9)] for _ in range(6)]

        self.par = {start: None}
        cost_map = {start: 0}
        self._add_compressed_point(start)
        path_found = False

        for its in range(max_iterations):
            if its % 250 == 0:
                logger.debug(
                    f'RRT info: ITERS: {its} nodes: {len(self.par.keys())}, BEST: {cost_map.get(goal, float("inf"))} EUCLID: {goal.distance(start)}'
                )
            if random.random() < self.EXPLORE_BIAS:
                rand_point = Point(
                    random.uniform(-Field.HALF_LENGTH, Field.HALF_LENGTH),
                    random.uniform(-Field.HALF_WIDTH, Field.HALF_WIDTH),
                )
            else:
                rand_point = Point(target[0], target[1])

            closest_point = min(self.par.keys(), key=lambda p: p.distance(rand_point))
            new_segment = LineString([closest_point, rand_point])
            rand_point = new_segment.interpolate(self.STEP_SIZE)
            new_segment = LineString([closest_point, rand_point])

            if (
                self._closest_obstacle(friendly_robot_id, new_segment)
                > self.SAFE_OBSTACLES_RADIUS
                and rand_point not in self.par
            ):
                # Choose the best parent node
                best_parent = closest_point
                min_cost = cost_map[closest_point] + closest_point.distance(rand_point)
                for p in self._get_nearby(rand_point):
                    if (
                        cost_map[p] + p.distance(rand_point) < min_cost
                        and self._closest_obstacle(
                            friendly_robot_id, LineString([p, rand_point])
                        )
                        > self.SAFE_OBSTACLES_RADIUS
                    ):
                        best_parent = p
                        min_cost = cost_map[p] + p.distance(rand_point)

                cost_map[rand_point] = min_cost
                self.par[rand_point] = best_parent

                # Now need to rewire the tree based on this

                for p in self._get_nearby(rand_point):
                    # Might be able to find a better path through rand_point
                    if (
                        cost_map[rand_point] + rand_point.distance(p) < cost_map[p]
                        and self._closest_obstacle(
                            friendly_robot_id, LineString([p, rand_point])
                        )
                        > self.SAFE_OBSTACLES_RADIUS
                    ):
                        cost_map[p] = cost_map[rand_point] + rand_point.distance(p)
                        self.par[p] = rand_point
                        self._propagate(self.par, cost_map, p)

                if goal.distance(
                    LineString([best_parent, rand_point])
                ) < self.STOPPING_DISTANCE and cost_map[
                    rand_point
                ] + rand_point.distance(
                    goal
                ) < cost_map.get(
                    goal, float("inf")
                ):
                    self.par[goal] = rand_point
                    cost_map[goal] = cost_map[rand_point] + rand_point.distance(goal)
                    path_found = True
                    if (
                        cost_map[goal] <= self.GOOD_ENOUGH_ABS * goal.distance(start)
                        or cost_map[goal] - goal.distance(start) < self.GOOD_ENOUGH_ABS
                    ):
                        break
                self._add_compressed_point(rand_point)

            if (
                cost_map.get(goal, float("inf"))
                <= self.GOOD_ENOUGH_ABS * goal.distance(start)
                or cost_map.get(goal, float("inf")) - goal.distance(start)
                < self.GOOD_ENOUGH_ABS
            ):
                break

            if its >= 1000 and path_found:
                break

        if path_found:
            path = []
            current_point = goal
            visited = set()
            while current_point is not None:
                if current_point in visited:
                    # There was a cycle in the tree - this is very bad
                    logger.warning("RRT Planner: Cycle in the tree - this is very bad")
                    return None
                visited.add(current_point)
                path.append(current_point)
                current_point = self.par[current_point]
            path.reverse()
            return self._reduce_waypoints([(p.x, p.y) for p in path])

        return None


N_DIRECTIONS = 16


def intersects_any_polygon(segment: LineString, obstacles: List[Polygon]) -> bool:
    for o in obstacles:
        if segment.distance(o.boundary) < ROBOT_RADIUS:
            return True
    return False


class DynamicWindowPlanner:
    """This class is a stateless planning class and should not be used on its own
    see the controllers class which provide state tracking and waypoint switching for these classes such as TimedSwitchController
    """
    SIMULATED_TIMESTEP = 0.2  # seconds
    MAX_ACCELERATION = 2  # Measured in ms^2
    DIRECTIONS = [i * 2 * pi / N_DIRECTIONS for i in range(N_DIRECTIONS)]
    ROBOT_RADIUS = 0.1

    def __init__(self, game: Game):
        self._game = game
        self._friendly_colour = Colour.YELLOW if game.my_team_is_yellow else Colour.Blue

    def path_to(
        self,
        friendly_robot_id: int,
        target: Tuple[float, float],
        temporary_obstacles: List[LineString] = [],
    ) -> Tuple[Tuple[float, float], float]:
        """
        Plan a path to the target for the specified friendly robot.

        Args:
            friendly_robot_id (int): The ID of the friendly robot.
            target (Tuple[float, float]): The target coordinates (x, y).

        Returns:
            Tuple[float, float]: The next waypoint coordinates (x, y) or the target if already reached.
        """
        robot: Robot = self._game.friendly_robots[friendly_robot_id]
        start_x, start_y = robot.x, robot.y

        if dist((start_x, start_y), target) < 1.5 * ROBOT_RADIUS:
            return target, float("inf")

        return self.local_planning(friendly_robot_id, target, temporary_obstacles)

    def local_planning(
        self,
        friendly_robot_id: int,
        target: Tuple[float, float],
        temporary_obstacles: List[Polygon],
    ) -> Tuple[Tuple[float, float], float]:
        velocity = self._game.get_object_velocity(
            GameRobot(self._friendly_colour, friendly_robot_id)
        )

        if velocity is None:
            # If no data, assume it is still :)
            velocity = 0, 0

        # Calculate the allowed velocities in this frame
        delta_vel = (
            DynamicWindowPlanner.SIMULATED_TIMESTEP
            * DynamicWindowPlanner.MAX_ACCELERATION
        )
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
                segment = self._get_motion_segment(
                    (start_x, start_y), velocity, delta_vel * sf, ang
                )
                if intersects_any_polygon(segment, temporary_obstacles):
                    continue
                # Evaluate this segment, avoiding obstacles
                score = self._evaluate_segment(
                    friendly_robot_id, segment, Point(target[0], target[1])
                )

                if score > best_score:
                    best_score = score
                    best_move = segment.coords[1]

            sf /= 4

        return best_move, best_score

    def _get_obstacles(self, robot_id):
        return (
            self._game.friendly_robots[:robot_id]
            + self._game.friendly_robots[robot_id + 1 :]
            + self._game.enemy_robots
        )

    def make_inf_long(self, segment: LineString):
        norm = segment.length
        endX, endY = segment.coords[1]
        startX, startY = segment.coords[0]
        new = Point(
            startX + (endX - startX) / norm * 18, startY + (endY - startY) / norm * 18
        )
        return LineString([segment.coords[0], new])

    def obstacle_penalty_function(self, x):
        return exp(-8 * (x - 0.22))

    def target_closeness_function(self, x):
        return 4 * exp(-8 * x)

    def _evaluate_segment(
        self, robot_id: int, segment: LineString, target: Point
    ) -> float:
        """Evaluate line segment; bigger score is better"""
        # Distance travelled towards the target should be rewarded
        target_factor = target.distance(Point(segment.coords[0])) - target.distance(
            Point(segment.coords[1])
        )
        our_velocity_vector = (
            segment.coords[1][0] - segment.coords[0][0]
        ) / self.SIMULATED_TIMESTEP, (
            segment.coords[1][1] - segment.coords[0][1]
        ) / self.SIMULATED_TIMESTEP
        if our_velocity_vector is None:
            our_velocity_vector = (0, 0)

        our_position = self._game.get_robot_pos(
            self._friendly_colour == Colour.YELLOW, robot_id
        )

        # If we are too close to an obstacle, we should be penalised
        obstacle_factor = 0

        # Factor in obstacle velocity, do some maths to find their closest approach to us
        # and the time at which that happens.
        for r in self._get_obstacles(robot_id):
            their_velocity_vector = self._game.get_object_velocity(
                GameRobot(
                    (
                        self._friendly_colour
                        if r.is_friendly
                        else Colour.invert(self._friendly_colour)
                    ),
                    r.id,
                )
            )

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
                    d_sq = (diff_p_x + t * diff_v_x) ** 2 + (
                        diff_p_y + t * diff_v_y
                    ) ** 2

                    obstacle_factor = max(
                        obstacle_factor,
                        self.obstacle_penalty_function(d_sq)
                        * self.obstacle_penalty_function(t),
                    )

        # Adjust weights for the final score - this is done by tuning
        score = (
            5 * target_factor
            - obstacle_factor
            + self.target_closeness_function(target.distance(segment))
        )
        return score

    def _get_motion_segment(
        self,
        rpos: Tuple[float, float],
        rvel: Tuple[float, float],
        delta_vel: float,
        ang: float,
    ) -> LineString:
        adj_vel_y = rvel[1] * DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel * sin(
            ang
        )
        adj_vel_x = rvel[0] * DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel * cos(
            ang
        )
        end_y = adj_vel_y + rpos[1]
        end_x = adj_vel_x + rpos[0]
        return LineString([(rpos[0], rpos[1]), (end_x, end_y)])


class BisectorPlanner:
    """This class is a stateless planning class and should not be used on its own
    see the controllers class which provide state tracking and waypoint switching for these classes such as TimedSwitchController
    """
    OBSTACLE_CLEARANCE = ROBOT_DIAMETER
    ClOSE_LIMIT = 0.5
    SAMPLE_SIZE = 0.10

    def _get_obstacles(self, robot_id):
        return (
            self._game.friendly_robots[:robot_id]
            + self._game.friendly_robots[robot_id + 1 :]
            + self._game.enemy_robots
        )

    def __init__(self, game, friendly_colour, env):
        self._game = game
        self._friendly_colour = friendly_colour
        self._enemy_colour = Colour.invert(friendly_colour)
        self._env = env

    def perpendicular_bisector(self, line: LineString):
        # Ensure the input is a valid LineString
        if len(line.coords) != 2:
            raise ValueError("The LineString must consist of exactly two points.")

        # Get the midpoint of the line
        midpoint = line.interpolate(0.5, normalized=True)

        # Rotate the line by 90 degrees around the midpoint
        perpendicular = rotate(line, 90, origin=midpoint)
        sx, sy = perpendicular.coords[0]
        ex, ey = perpendicular.coords[1]

        gradx = ex - sx
        grady = ey - sy

        start_x = sx - 1000 * gradx
        start_y = sy - 1000 * grady

        end_x = ex + 1000 * gradx
        end_y = ey + 1000 * grady

        return LineString([(start_x, start_y), (end_x, end_y)])

    def _adjust_segment_for_robot_radius(self, seg: LineString) -> LineString:
        current_robot_seg_interect = seg.interpolate(ROBOT_RADIUS)
        return LineString(
            [current_robot_seg_interect, seg.interpolate(1, normalized=True)]
        )

    def path_to(
        self,
        robot_id: int,
        target: Tuple[float, float],
        temporary_obstacles: List[Polygon] = [],
    ) -> Tuple[float, float]:
        """
        Calculate a path for the robot to the target position while avoiding obstacles.
        Args:
            robot_id (int): The ID of the robot for which the path is being calculated.
            target (Tuple[float, float]): The target position (x, y) to which the robot should move.
            temporary_obstacles (List[Polygon]): A list of temporary obstacles represented as Polygon objects.
                These obstacles represent imaginary and temporary regions to avoid, such as defense areas during play.
                During setup time and ball placement, the robot may be allowed to enter these areas. For temporary obstacles
                we assume that entering them is possible but not desirable. 
        Returns:
            Tuple[float, float]: The next position (x, y) for the robot to move towards the target.
        """

        our_position = self._game.get_robot_pos(
            self._friendly_colour == self._friendly_colour, robot_id
        )

        if self._env is not None:
            self._env.draw_line(
                [(our_position.x, our_position.y), target], width=2, color="GREEN"
            )

        line = LineString([(our_position.x, our_position.y), target])

        # Stops jittering near the target
        if line.length < BisectorPlanner.ClOSE_LIMIT:
            return target

        perp = self.perpendicular_bisector(line)

        midpoint = perp.interpolate(0.5, normalized=True)

        if midpoint.distance(Point(*target)) < ROBOT_RADIUS:
            return target

        halves = [
            LineString([midpoint, perp.coords[1]]),
            LineString([midpoint, perp.coords[0]]),
        ]
        obsts = self._get_obstacles(robot_id)

        if self._env is not None:
            self._env.draw_line(halves[0].coords, width=3)
            self._env.draw_line(halves[1].coords, width=3)

        for s in range(
            int(
                max(Field.HALF_LENGTH * 2, Field.HALF_WIDTH * 2)
                / BisectorPlanner.SAMPLE_SIZE
            )
        ):
            offset = s * BisectorPlanner.SAMPLE_SIZE

            for h in halves:
                p1 = h.interpolate(offset)

                seg1 = self._adjust_segment_for_robot_radius(
                    LineString([(our_position.x, our_position.y), p1])
                )
                seg2 = LineString([p1, target])

                if not self._segment_intersects(
                    seg1, obsts
                ) and not self._segment_intersects(seg2, obsts):

                    if self._env is not None:

                        self._env.draw_point(*point_to_tuple(p1), color="PINK", width=1)
                        col = "GREEN" if not intersects_any_polygon(seg1, temporary_obstacles) else "RED"
                        self._env.draw_line(list(seg1.coords), width=1, color=col)
                        self._env.draw_line(list(seg2.coords), width=3, color="PINK")
                    if not intersects_any_polygon(
                        seg1, temporary_obstacles
                    ) and not intersects_any_polygon(seg2, temporary_obstacles):
                        return point_to_tuple(p1)

        return point_to_tuple(midpoint)

    def _segment_intersects(
        self, seg: LineString, obstacles: List[Tuple[float, float]]
    ) -> bool:

        for o in obstacles:
            if Point((o.x, o.y)).distance(seg) < BisectorPlanner.OBSTACLE_CLEARANCE:
                return True
        return False
