import copy
import math
from math import exp
from typing import Iterable, List, Optional

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
        config: DynamicWindowConfig | None = None,
        env: SSLStandardEnv | None = None,
    ):
        self._config = config or DynamicWindowConfig()
        self._simulate_timestep = self._config.simulate_frames * TIMESTEP
        self._control_period = TIMESTEP
        self._max_acceleration = self._config.max_acceleration
        self._max_safety_radius = self._config.max_safety_radius
        self._safety_penalty_distance_sq = self._config.safety_penalty_distance_sq
        self._max_speed_for_full_bubble = self._config.max_speed_for_full_bubble
        self.env = env

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
        velocity = robot.v
        current_speed = velocity.mag()
        safety_radius = self._dynamic_safety_radius(current_speed)
        safety_radius_sq = safety_radius * safety_radius

        delta_max_vel = self._control_period * self._max_acceleration
        best_score = float("-inf")
        start = robot.p
        best_move = start

        rect_obstacles = to_rectangles(temporary_obstacles)

        dx, dy = target - start
        ang0 = math.atan2(dy, dx)
        step = 2 * math.pi / self._config.n_directions
        ordered_angles = [normalise_heading(ang0 + k * step) for k in range(self._config.n_directions)]

        for scale in self._candidate_scales():
            for ang in ordered_angles:
                segment_start, segment_end = self._get_motion_segment(
                    start,
                    velocity,
                    delta_max_vel * scale,
                    ang,
                )

                if self.env is not None:
                    self.env.draw_line([segment_start, segment_end], color="red", width=2)

                if self._segment_too_close(segment_start, segment_end, rect_obstacles, safety_radius):
                    continue

                score = self._evaluate_segment(
                    game,
                    robot,
                    segment_start,
                    segment_end,
                    target,
                    safety_radius_sq,
                )

                if score > best_score:
                    best_score = score
                    best_move = segment_end

            ### EARLY EXIT ###
            ## NEEDS TO BE CAREFUL WITH THIS TO AVOID SUBOPTIMAL PATHS ##
            if best_score >= 0:
                break

        if best_score == float("-inf"):
            return None

        segment_start = copy.copy(start)
        if self._segment_overshoots_target(segment_start, best_move, target):
            best_move = copy.copy(target)

        return best_move, best_score

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
