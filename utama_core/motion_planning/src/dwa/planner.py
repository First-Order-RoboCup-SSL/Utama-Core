from math import cos, exp, sin
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from utama_core.config.settings import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game, Robot

ROBOT_DIAMETER = 2 * ROBOT_RADIUS
N_DIRECTIONS = 16


def intersects_any_polygon(segment: LineString, obstacles: List[Polygon]) -> bool:
    # Buffer the segment by the robot radius to account for the robot's size
    seg_buffer = segment.buffer(ROBOT_RADIUS)
    for o in obstacles:
        if seg_buffer.intersects(o):
            return True
    return False


class DynamicWindowPlanner:
    SIMULATED_TIMESTEP = 1 / 60  # seconds
    MAX_ACCELERATION = 2  # Measured in m/s^2
    DIRECTIONS = [i * 2 * np.pi / N_DIRECTIONS for i in range(N_DIRECTIONS)]

    def __init__(self):
        self._game: Game = None

    def path_to(
        self,
        game: Game,
        friendly_robot_id: int,
        target: Vector2D,
    ) -> Tuple[Vector2D, float]:
        self._game = game
        robot: Robot = self._game.friendly_robots[friendly_robot_id]

        if robot.p.distance_to(target) < 1.1 * ROBOT_RADIUS:
            return target, float("inf")

        return self.local_planning(friendly_robot_id, target, [])

    def local_planning(
        self,
        friendly_robot_id: int,
        target: Vector2D,
        temporary_obstacles: List[Polygon],
    ) -> Tuple[Vector2D, float]:
        robot: Robot = self._game.friendly_robots[friendly_robot_id]
        velocity = robot.v

        delta_vel = self.SIMULATED_TIMESTEP * self.MAX_ACCELERATION
        best_score = float("-inf")
        best_move = robot.p

        sf = 1
        while best_score < 0 and sf > 0.05:
            for ang in self.DIRECTIONS:
                segment = self._get_motion_segment(robot.p, velocity, delta_vel * sf, ang)
                if intersects_any_polygon(segment, temporary_obstacles):
                    continue
                score = self._evaluate_segment(friendly_robot_id, segment, target)
                if score > best_score:
                    best_score = score
                    best_move = Vector2D(segment.coords[1][0], segment.coords[1][1])
            sf /= 4

        dist_to_target = robot.p.distance_to(target)
        slowdown_scale = min(1.0, dist_to_target / 3)
        best_vel = Vector2D(
            (best_move.x - robot.p.x) / self.SIMULATED_TIMESTEP * slowdown_scale,
            (best_move.y - robot.p.y) / self.SIMULATED_TIMESTEP * slowdown_scale,
        )
        return best_vel, best_score

    def _get_obstacles(self, robot_id: int) -> List[Robot]:
        return [r for r in self._game.friendly_robots.values() if r.id != robot_id] + list(
            self._game.enemy_robots.values()
        )

    def obstacle_penalty_function(self, x: float) -> float:
        return exp(-8 * (x - 0.22))

    def target_closeness_function(self, x: float) -> float:
        return 4 * exp(-8 * x)

    def _evaluate_segment(self, robot_id: int, segment: LineString, target: Vector2D) -> float:
        start_point = Vector2D(*segment.coords[0])
        end_point = Vector2D(*segment.coords[1])

        # Distance travelled towards the target
        target_factor = start_point.distance_to(target) - end_point.distance_to(target)

        our_velocity_vector = Vector2D(
            (end_point.x - start_point.x) / self.SIMULATED_TIMESTEP,
            (end_point.y - start_point.y) / self.SIMULATED_TIMESTEP,
        )

        obstacle_factor = 0
        our_position = self._game.friendly_robots[robot_id].p

        for r in self._get_obstacles(robot_id):
            diff_v = Vector2D(our_velocity_vector.x - r.v.x, our_velocity_vector.y - r.v.y)
            diff_p = Vector2D(our_position.x - r.p.x, our_position.y - r.p.y)

            denom = diff_v.x**2 + diff_v.y**2
            if denom != 0:
                t = -(diff_v.x * diff_p.x + diff_v.y * diff_p.y) / denom
                if t > 0:
                    d_sq = (diff_p.x + t * diff_v.x) ** 2 + (diff_p.y + t * diff_v.y) ** 2
                    obstacle_factor = max(
                        obstacle_factor,
                        self.obstacle_penalty_function(d_sq) * self.obstacle_penalty_function(t),
                    )

        score = 5 * target_factor - obstacle_factor + self.target_closeness_function(end_point.distance_to(target))
        return score

    def _get_motion_segment(self, rpos: Vector2D, rvel: Vector2D, delta_vel: float, ang: float) -> LineString:
        adj_vel = Vector2D(rvel.x + delta_vel * cos(ang), rvel.y + delta_vel * sin(ang))
        end_pos = Vector2D(
            rpos.x + adj_vel.x * self.SIMULATED_TIMESTEP,
            rpos.y + adj_vel.y * self.SIMULATED_TIMESTEP,
        )
        return LineString([(rpos.x, rpos.y), (end_pos.x, end_pos.y)])
