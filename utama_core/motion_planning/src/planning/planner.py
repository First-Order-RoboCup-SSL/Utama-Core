from math import cos, dist, exp, sin
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from utama_core.config.settings import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game, Robot

ROBOT_DIAMETER = 2 * ROBOT_RADIUS
N_DIRECTIONS = 16


def intersects_any_polygon(segment: LineString, obstacles: List[Polygon]) -> bool:
    for o in obstacles:
        if segment.distance(o.boundary) < ROBOT_RADIUS:
            return True
    return False


class DynamicWindowPlanner:
    """This class is a stateless planning class and should not be used on its own
    see the controllers class which provide state tracking and waypoint switching for these classes such as TimedSwitchController

    TODO Add code to leave the defense areas via the nearest point such that the robot is outside the polygon
    this should not be in the planners as we need this to be more reactive, happening during any skills
    like go_to_ball. The behaviour tree should handle this.


    """

    SIMULATED_TIMESTEP = 1 / 60  # seconds
    MAX_ACCELERATION = 2  # Measured in ms^2
    DIRECTIONS = [i * 2 * np.pi / N_DIRECTIONS for i in range(N_DIRECTIONS)]

    def __init__(self):
        self._game = None

    def path_to(
        self,
        game: Game,
        friendly_robot_id: int,
        target: Vector2D,
    ) -> Tuple[Vector2D, float]:
        """
        Plan a path to the target for the specified friendly robot.

        Args:
            friendly_robot_id (int): The ID of the friendly robot.
            target (Vector2D): The target coordinates (x, y).

        Returns:
            Tuple[Vector2D, float]: The next waypoint coordinates (x, y) or the target if already reached and score.
        """
        temporary_obstacles: List[Polygon] = []
        self._game = game
        robot: Robot = self._game.friendly_robots[friendly_robot_id]

        if robot.p.distance_to(target) < 1.1 * ROBOT_RADIUS:
            return target, float("inf")

        return self.local_planning(friendly_robot_id, target, temporary_obstacles)

    def local_planning(
        self,
        friendly_robot_id: int,
        target: Vector2D,
        temporary_obstacles: List[Polygon],
    ) -> Tuple[Vector2D, float]:
        velocity = self._game.friendly_robots[friendly_robot_id].v

        # Calculate the allowed velocities in this frame
        delta_vel = DynamicWindowPlanner.SIMULATED_TIMESTEP * DynamicWindowPlanner.MAX_ACCELERATION
        best_score = float("-inf")
        robot: Robot = self._game.friendly_robots[friendly_robot_id]

        best_move = robot.p.x, robot.p.y

        # sf is the scale factor for the velocity - we start with full velocity to prioritise speed
        # and then reduce it if we can't find a good segment, this allows the robot to dynamically adjust
        # its speed and path length to avoid obstacles
        sf = 1
        while best_score < 0 and sf > 0.05:
            for ang in DynamicWindowPlanner.DIRECTIONS:
                segment = self._get_motion_segment(robot.p, velocity, delta_vel * sf, ang)
                if intersects_any_polygon(segment, temporary_obstacles):
                    continue
                # Evaluate this segment, avoiding obstacles
                score = self._evaluate_segment(friendly_robot_id, segment, Point(target.x, target.y))

                if score > best_score:
                    best_score = score
                    best_move = segment.coords[1]

            sf /= 4
        dist_to_target = robot.p.distance_to(target)
        slowdown_scale = min(1.0, dist_to_target / 3)
        best_vel = (
            (best_move[0] - robot.p.x) / DynamicWindowPlanner.SIMULATED_TIMESTEP * slowdown_scale,
            (best_move[1] - robot.p.y) / DynamicWindowPlanner.SIMULATED_TIMESTEP * slowdown_scale,
        )
        return best_vel, best_score

    def _get_obstacles(self, robot_id: int) -> List[Robot]:
        return [robot for robot in self._game.friendly_robots.values() if robot.id != robot_id] + [
            robot for robot in self._game.enemy_robots.values()
        ]

    def obstacle_penalty_function(self, x):
        return exp(-8 * (x - 0.22))

    def target_closeness_function(self, x):
        return 4 * exp(-8 * x)

    def _evaluate_segment(self, robot_id: int, segment: LineString, target: Point) -> float:
        """Evaluate line segment; bigger score is better"""
        # Distance travelled towards the target should be rewarded
        target_factor = target.distance(Point(segment.coords[0])) - target.distance(Point(segment.coords[1]))
        our_velocity_vector = (
            (segment.coords[1][0] - segment.coords[0][0]) / self.SIMULATED_TIMESTEP,
            (segment.coords[1][1] - segment.coords[0][1]) / self.SIMULATED_TIMESTEP,
        )

        our_position = self._game.friendly_robots[robot_id].p

        # If we are too close to an obstacle, we should be penalised
        obstacle_factor = 0

        # Factor in obstacle velocity, do some maths to find their closest approach to us
        # and the time at which that happens.
        for r in self._get_obstacles(robot_id):
            their_velocity_vector = r.v
            their_position = r.p

            diff_v_x = our_velocity_vector[0] - their_velocity_vector[0]
            diff_p_x = our_position.x - their_position.x
            diff_v_y = our_velocity_vector[1] - their_velocity_vector[1]
            diff_p_y = our_position.y - their_position.y

            if (denom := (diff_v_x * diff_v_x + diff_v_y * diff_v_y)) != 0:
                t = (-diff_v_x * diff_p_x - diff_v_y * diff_p_y) / denom
                if t > 0:
                    d_sq = (diff_p_x + t * diff_v_x) ** 2 + (diff_p_y + t * diff_v_y) ** 2

                    obstacle_factor = max(
                        obstacle_factor,
                        self.obstacle_penalty_function(d_sq) * self.obstacle_penalty_function(t),
                    )

        # Adjust weights for the final score - this is done by tuning
        score = 5 * target_factor - obstacle_factor + self.target_closeness_function(target.distance(segment))
        return score

    def _get_motion_segment(
        self,
        rpos: Vector2D,
        rvel: Vector2D,
        delta_vel: float,
        ang: float,
    ) -> LineString:
        adj_vel_y = rvel[1] * DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel * sin(ang)
        adj_vel_x = rvel[0] * DynamicWindowPlanner.SIMULATED_TIMESTEP + delta_vel * cos(ang)
        end_y = adj_vel_y + rpos[1]
        end_x = adj_vel_x + rpos[0]
        return LineString([(rpos[0], rpos[1]), (end_x, end_y)])
