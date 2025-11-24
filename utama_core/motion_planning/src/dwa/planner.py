import copy
import math
from math import exp, hypot
from typing import Iterable, List, Optional, Tuple

import numpy as np

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game, Robot
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.motion_planning.src.dwa.config import DynamicWindowConfig
from utama_core.motion_planning.src.planning.geometry import (
    AxisAlignedRectangle,
    point_segment_distance,
)
from utama_core.motion_planning.src.planning.obstacles import (
    ObstacleRegion,
    to_rectangles,
)
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv


class DynamicWindowPlanner:
    """Stateless local planner backing the DWA translation controller."""

    def __init__(
        self,
        config: DynamicWindowConfig,
        env: SSLStandardEnv | None = None,
    ):
        self._config = config
        self._simulate_timestep = self._config.simulate_frames * TIMESTEP
        self._control_period = TIMESTEP
        self._max_acceleration = self._config.max_acceleration
        self._max_safety_radius = self._config.max_safety_radius
        self._safety_penalty_distance_sq = self._config.safety_penalty_distance_sq
        self._max_speed_for_full_bubble = self._config.max_speed_for_full_bubble
        self.env = env
        self._v_resolution = getattr(self._config, "v_resolution")
        self._w_goal = getattr(self._config, "weight_goal")
        self._w_obstacle = getattr(self._config, "weight_obstacle")
        self._w_speed = getattr(self._config, "weight_speed")

    def path_to(
        self,
        game: Game,
        friendly_robot_id: int,
        target: Vector2D,
        temporary_obstacles: Optional[List[ObstacleRegion]] = None,
    ) -> tuple[Vector2D, float] | None:
        """
        Plan a path for the specified friendly robot to the target position, avoiding any temporary obstacles.
        Args:
            friendly_robot_id (int): The ID of the friendly robot to plan for.
            target (Vector2D): The target position to move towards.
            temporary_obstacles (Optional[List[ObstacleRegion]]): A list of temporary obstacles to consider during planning.
        Returns:
            Optional[tuple[Vector2D, float]]: A tuple containing the best waypoint and its score,
            or None if no valid path is found.
        """
        robot = game.friendly_robots[friendly_robot_id]

        if robot.p.distance_to(target) < 1.5 * ROBOT_RADIUS:
            return target, float("inf")
        obstacles = temporary_obstacles or []
        return self._plan_local(game, robot, target, obstacles)

    def _plan_local(
        self,
        game: Game,
        robot: Robot,
        target: Vector2D,
        temporary_obstacles: List[ObstacleRegion],
    ) -> tuple[Vector2D, float] | None:
        """
        Plan a local motion segment towards the target while avoiding obstacles.
        Args:
            game (Game): The current game state.
            robot (Robot): The robot to plan for.
            target (Vector2D): The target position to move towards.
            temporary_obstacles (List[ObstacleRegion]): A list of temporary obstacles to consider during planning.
        Returns:
            Optional[tuple[Vector2D, float]]: A tuple containing the best waypoint and its score,
            or None if no valid path is found.
        """
        cur_v = robot.v
        rect_obstacles = to_rectangles(temporary_obstacles)
        start_pos = robot.p

        dw = self.calc_dynamic_window(cur_v)

        best_score = float("-inf")
        best_endpoint = start_pos

        vx_min, vx_max, vy_min, vy_max = dw

        vx_vals = np.arange(vx_min, vx_max + 1e-9, self._v_resolution)
        vy_vals = np.arange(vy_min, vy_max + 1e-9, self._v_resolution)

        # possibly optimise by sorting candidates by closeness to desired direction
        for vx in vx_vals:
            for vy in vy_vals:
                speed = hypot(vx, vy)
                if speed > self._config.max_speed + 1e-9:
                    continue

                traj = self.predict_trajectory(start_pos, vx, vy)

                obs_cost = self.calc_obstacle_cost(traj, rect_obstacles)
                goal_cost = self.calc_to_goal_cost(traj, target)
                speed_cost = self.calc_speed_cost(speed)

                score = self._w_goal * goal_cost - self._w_obstacle * obs_cost + self._w_speed * speed_cost
                if score > best_score:
                    best_score = score
                    best_endpoint = traj[-1] if len(traj) else start_pos

        if best_score == float("-inf"):  # best move is not ot move at all lol
            return None

        # clamp endpoint if overshooting
        if best_endpoint.distance_to(target) < self._config.target_tolerance:
            best_endpoint = target

        return best_endpoint, best_score

    def calc_dynamic_window(self, cur_v: Vector2D) -> Tuple[float, float, float, float]:
        """Computes dynamic window in (vx, vy) space
        - clamps velocities component-wise here BEFORE filtering in sampling loop"""

        dt = self._simulate_timestep
        a = self._max_acceleration

        vx_min = cur_v.x - a * dt
        vx_max = cur_v.x + a * dt
        vy_min = cur_v.y - a * dt
        vy_max = cur_v.y + a * dt

        vlimit = getattr(self._config, "max_speed")
        vx_min = max(vx_min, -vlimit)
        vx_max = min(vx_max, vlimit)
        vy_min = max(vy_min, -vlimit)
        vy_max = min(vy_max, vlimit)

        return vx_min, vx_max, vy_min, vy_max

    def predict_trajectory(
        self,
        start_pos: Vector2D,
        vx: float,
        vy: float,
    ) -> List[Vector2D]:
        """Simulates robot position
        - takes in candidate velocity components and current position
        - assumes constant velocity components (vx, vy)
        - creates a list of Vector2D positions along the trajectory
        - returns this trajectory list
        """

        steps = self._config.simulate_frames
        traj: List[Vector2D] = []
        pos = Vector2D(start_pos.x, start_pos.y)
        for i in range(steps):
            pos = Vector2D(pos.x + vx * TIMESTEP, pos.y + vy * TIMESTEP)
            traj.append(pos)
        return traj

    def calc_obstacle_cost(
        self,
        traj: List[Vector2D],
        rect_obstacles: List[AxisAlignedRectangle],
    ) -> float:
        """Compute a trajectory's obstacle cost
        - takes in trajectory (list of positions) and obstacle positions
        - higher cost if trajectory collides with obstacle
        - returns cost
        """

        if not traj:
            return 0.0

        min_dist_sq = float("inf")
        for i in range(len(traj) - 1):
            a = traj[i]
            b = traj[i + 1]
            for rect in rect_obstacles:
                d = rect.distance_to_segment(a, b)
                if d < ROBOT_RADIUS:  # make it ROBOT_RADIUS + <buffer> if extra "bubble" required
                    return float("inf")
                min_dist_sq = min(min_dist_sq, d**2)

        if min_dist_sq == float("inf"):
            return 0.0

        penalty = 0.0
        safe_dist_sq = getattr(self._config, "safe_dist_sq", 0.0625)  # default 0.0625
        if min_dist_sq < safe_dist_sq:
            penalty = exp(-(min_dist_sq / safe_dist_sq))
        return penalty

    def calc_to_goal_cost(
        self,
        traj: List[Vector2D],
        goal: Vector2D,
    ) -> float:
        """Compute reward for approaching goal
        - takes in trajectory (list of positions) and goal position
        - takes last position in trajectory list
        - greater reward for smaller distance to goal from ending position
        """

        if not traj:
            return 0.0
        end = traj[-1]
        dist = end.distance_to(goal)
        return 1.0 / (1.0 + dist)

    def calc_speed_cost(
        self,
        speed: float,
    ) -> float:
        """Compute reward for higher speed
        - reward = speed
        - NAIVE
        """

        return speed

    def _get_obstacles(self, game: Game, robot_id: int) -> List[Robot]:
        friendly = [r for rid, r in game.friendly_robots.items() if rid != robot_id]
        enemies = list(game.enemy_robots.values())
        return friendly + enemies

    def _obstacle_penalty(self, value: float) -> float:
        return exp(-8 * (value - self._safety_penalty_distance_sq))

    @staticmethod
    def _target_closeness(value: float) -> float:
        return 4 * exp(-8 * value)

    def _evaluate_segment(
        self,
        game: Game,
        robot: Robot,
        start: Vector2D,
        end: Vector2D,
        target: Vector2D,
        safety_radius_sq: float,
    ) -> float:
        """Evaluate a candidate motion segment; higher score is better."""
        seg_vec = end - start

        start_dist = target.distance_to(start)
        end_dist = target.distance_to(end)
        target_factor = start_dist - end_dist

        our_velocity_vector = seg_vec / self._simulate_timestep
        our_position = robot.p

        obstacle_factor = 0.0
        for obstacle in self._get_obstacles(game, robot.id):
            their_velocity = obstacle.v
            their_position = obstacle.p

            diff_v = our_velocity_vector - their_velocity
            denom = diff_v.dot(diff_v)
            if denom == 0.0:
                continue

            diff_p = our_position - their_position
            t = -diff_v.dot(diff_p) / denom
            if t <= 0.0:
                continue

            closest = diff_p + t * diff_v
            d_sq = closest.dot(closest)
            adjustment = max(self._safety_penalty_distance_sq - safety_radius_sq, 0.0)
            effective_d_sq = d_sq + adjustment
            obstacle_factor = max(
                obstacle_factor,
                self._obstacle_penalty(effective_d_sq) * self._obstacle_penalty(t),
            )

        distance_to_line = point_segment_distance(target, start, end)
        score = 5 * target_factor - obstacle_factor + self._target_closeness(distance_to_line)
        return score

    def _dynamic_safety_radius(self, speed: float) -> float:
        """Interpolate the clearance radius between the physical robot radius and the nominal DWA bubble."""
        min_radius = ROBOT_RADIUS
        max_radius = self._max_safety_radius
        if max_radius <= min_radius:
            return max_radius
        if self._max_speed_for_full_bubble <= 1e-6:
            return max_radius
        ratio = min(max(speed / self._max_speed_for_full_bubble, 0.0), 1.0)
        return min_radius + (max_radius - min_radius) * ratio

    def _get_motion_segment(
        self,
        rpos: Vector2D,
        rvel: Vector2D,
        delta_max_vel: float,
        ang: float,
    ) -> tuple[Vector2D, Vector2D]:
        direction = Vector2D(math.cos(ang), math.sin(ang))
        new_velocity = rvel + delta_max_vel * direction
        displacement = new_velocity * self._simulate_timestep
        end = rpos + displacement
        return rpos, end

    def _candidate_scales(self) -> Iterable[float]:
        """Yield velocity scale factors from fast to slow until the window shrinks."""
        scale = 1.0
        while scale > 0.05:
            yield scale
            scale /= 4

    def _segment_overshoots_target(self, start: Vector2D, end: Vector2D, target: Vector2D) -> bool:
        """Return True when the candidate segment would pass through the target."""
        segment = end - start
        seg_len_sq = segment.dot(segment)
        if seg_len_sq <= 1e-9:
            return False

        to_target = target - start
        projection = to_target.dot(segment)
        if projection <= 0.0 or projection >= seg_len_sq:
            return False

        distance = point_segment_distance(target, start, end)
        return distance <= self._config.target_tolerance

    def _segment_too_close(
        self,
        start: Vector2D,
        end: Vector2D,
        obstacles: List[AxisAlignedRectangle],
        clearance: float,
    ) -> bool:
        for obstacle in obstacles:
            if obstacle.distance_to_segment(start, end) < clearance:
                return True
        return False
