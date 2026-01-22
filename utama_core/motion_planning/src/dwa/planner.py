import copy
import math
import random
from typing import List, Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game, Robot
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.motion_planning.src.dwa.config import DynamicWindowConfig
from utama_core.motion_planning.src.planning.geometry import point_segment_distance
from utama_core.motion_planning.src.planning.obstacles import ObstacleRegion
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv


class OrientedRectangle:
    """Oriented bounding box for velocity-based safety area."""

    def __init__(self, center: Vector2D, width: float, length: float, heading: float):
        """
        Args:
            center: Center position of the rectangle
            width: Width perpendicular to heading (left + right clearance)
            length: Length along heading direction (forward clearance)
            heading: Orientation in radians (direction robot is moving)
        """
        self.center = center
        self.width = width
        self.length = length
        self.heading = heading

        # Precompute rotation
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        self.cos_heading = cos_h
        self.sin_heading = sin_h

        # Half extents
        self.half_width = width / 2.0
        self.half_length = length / 2.0

    def get_corners(self) -> List[Vector2D]:
        """Get the four corners of the rectangle in world coordinates."""
        # Local coordinates (centered at origin, aligned with x-axis)
        # Forward is +x, left is +y
        local_corners = [
            Vector2D(self.half_length, self.half_width),  # Front-left
            Vector2D(self.half_length, -self.half_width),  # Front-right
            Vector2D(-self.half_length, -self.half_width),  # Back-right
            Vector2D(-self.half_length, self.half_width),  # Back-left
        ]

        # Rotate and translate to world coordinates
        world_corners = []
        for local in local_corners:
            # Rotate
            rotated_x = local.x * self.cos_heading - local.y * self.sin_heading
            rotated_y = local.x * self.sin_heading + local.y * self.cos_heading
            # Translate
            world = Vector2D(self.center.x + rotated_x, self.center.y + rotated_y)
            world_corners.append(world)

        return world_corners

    def intersects(self, other: "OrientedRectangle") -> bool:
        """
        Check if this rectangle intersects with another using Separating Axis Theorem (SAT).

        Returns:
            True if rectangles overlap, False otherwise
        """
        # Get corners of both rectangles
        corners_a = self.get_corners()
        corners_b = other.get_corners()

        # Test separation along axes perpendicular to each rectangle's edges
        # For rectangles, we need to test 4 axes (2 per rectangle)
        axes = [
            Vector2D(self.cos_heading, self.sin_heading),  # This rect's forward axis
            Vector2D(-self.sin_heading, self.cos_heading),  # This rect's side axis
            Vector2D(other.cos_heading, other.sin_heading),  # Other rect's forward axis
            Vector2D(-other.sin_heading, other.cos_heading),  # Other rect's side axis
        ]

        for axis in axes:
            # Project all corners onto this axis
            proj_a = [corner.dot(axis) for corner in corners_a]
            proj_b = [corner.dot(axis) for corner in corners_b]

            # Find min/max projections
            min_a, max_a = min(proj_a), max(proj_a)
            min_b, max_b = min(proj_b), max(proj_b)

            # Check for separation on this axis
            if max_a < min_b or max_b < min_a:
                return False  # Separating axis found, no intersection

        # No separating axis found, rectangles must intersect
        return True

    def distance_to(self, other: "OrientedRectangle") -> float:
        """
        Approximate distance between two oriented rectangles.
        Returns 0 if they intersect, otherwise approximate minimum distance.
        """
        if self.intersects(other):
            return 0.0

        # Simple approximation: distance between centers minus half-extents
        center_dist = self.center.distance_to(other.center)

        # Rough estimate of closest approach
        # (not exact, but sufficient for penalty calculation)
        approx_dist = center_dist - (self.half_length + other.half_length) / 2.0
        return max(0.0, approx_dist)

    def intersection_ratio(self, other: "OrientedRectangle") -> float:
        """
        Calculate the intersection ratio between two rectangles.

        Returns:
            A value between 0 and 1 representing the degree of overlap:
            - 0.0: No intersection
            - 0.5: Moderate overlap
            - 1.0: Complete overlap (one rectangle fully inside the other)
        """
        if not self.intersects(other):
            return 0.0

        # Approximate intersection area using center distance and dimensions
        # This is a simplified calculation that gives reasonable results
        center_dist = self.center.distance_to(other.center)

        # Maximum possible distance for complete separation
        max_separation = math.sqrt(
            (self.half_length + other.half_length) ** 2 + (self.half_width + other.half_width) ** 2
        )

        # If centers are very close, high overlap
        # If centers are at max_separation, minimal overlap (just touching)
        if max_separation < 1e-6:
            return 1.0

        # Normalized overlap: 1.0 when centers coincide, 0.0 at max separation
        overlap_ratio = 1.0 - min(center_dist / max_separation, 1.0)

        # Apply non-linear scaling to emphasize dangerous overlaps
        # This makes small overlaps less severe, large overlaps very severe
        intersection_score = overlap_ratio**1.5

        return min(intersection_score, 1.0)

    def contains_point(self, point: Vector2D) -> bool:
        """
        Check if a point is inside this oriented rectangle.

        Args:
            point: The point to test

        Returns:
            True if the point is inside the rectangle, False otherwise
        """
        # Transform point to local rectangle coordinates
        # Translate to origin
        dx = point.x - self.center.x
        dy = point.y - self.center.y

        # Rotate by -heading to align with rectangle's local axes
        local_x = dx * self.cos_heading + dy * self.sin_heading
        local_y = -dx * self.sin_heading + dy * self.cos_heading

        # Check if within bounds
        return abs(local_x) <= self.half_length and abs(local_y) <= self.half_width


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
        self.env = env
        self._w_goal = getattr(self._config, "weight_goal")
        self._w_obstacle = getattr(self._config, "weight_obstacle")
        self._w_speed = getattr(self._config, "weight_speed")

        # Safety area dimensions (configurable)
        self._side_clearance = 0.15  # Clearance on left/right sides (m)
        self._back_clearance = 0.12  # Clearance behind robot (m)
        self._base_front_clearance = 0.2  # Minimum forward clearance at zero speed (m)
        self._forward_lookahead_time = 0.5  # How many seconds to look ahead (s)

    def _create_safety_rectangle(self, position: Vector2D, velocity: Vector2D) -> OrientedRectangle:
        """
        Create a velocity-oriented rectangular safety area.

        Args:
            position: Current position of the robot
            velocity: Current velocity vector

        Returns:
            OrientedRectangle representing the safety area
        """
        speed = velocity.mag()

        # Determine heading from velocity
        if speed > 0.01:
            heading = math.atan2(velocity.y, velocity.x)
        else:
            # Stationary robot: use arbitrary heading or last known
            heading = 0.0

        # Width: fixed clearance on both sides
        width = 2 * ROBOT_RADIUS + 2 * self._side_clearance

        # Length: back clearance + front clearance (speed-dependent)
        # Forward extension = base + speed × lookahead_time
        forward_extension = self._base_front_clearance + speed * self._forward_lookahead_time
        total_length = self._back_clearance + ROBOT_RADIUS + forward_extension

        # Center the rectangle: shift forward by (forward - back) / 2
        # In local frame: center is at (forward - back) / 2
        local_offset = (forward_extension - self._back_clearance) / 2.0
        center = Vector2D(position.x + local_offset * math.cos(heading), position.y + local_offset * math.sin(heading))

        return OrientedRectangle(center, width, total_length, heading)

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

        delta_max_vel = self._control_period * self._max_acceleration
        best_score = float("-inf")
        start = robot.p
        best_move = start

        dx, dy = target - start
        ang0 = math.atan2(dy, dx)
        scales = self._random_candidate_scales(start.distance_to(target))

        # check the direction facing the target first
        best_collides = False
        for scale in scales:
            segment_start, segment_end = self._get_motion_segment(
                start,
                velocity,
                delta_max_vel * scale,
                ang0,
            )

            if self.env is not None:
                self.env.draw_line([segment_start, segment_end], color="red", width=2)

            score = self._evaluate_segment(game, robot, segment_start, segment_end, target)
            collides = self._segment_collides(game, robot, segment_start, segment_end)

            if score > best_score:
                best_score = score
                best_move = segment_end
                best_collides = collides

        if best_collides:
            for ang in self._random_candidate_angles(ang0, 5):
                for scale in scales:
                    segment_start, segment_end = self._get_motion_segment(
                        start,
                        velocity,
                        delta_max_vel * scale,
                        ang,
                    )

                    if self.env is not None:
                        self.env.draw_line([segment_start, segment_end], color="red", width=2)

                    score = self._evaluate_segment(game, robot, segment_start, segment_end, target)

                    if score > best_score:
                        best_score = score
                        best_move = segment_end

        if best_score == float("-inf"):  # best move is not to move at all lol
            return None

        if self._segment_overshoots_target(start, best_move, target):
            best_move = copy.copy(target)

        return best_move, best_score

    def _get_obstacles(self, game: Game, robot_id: int) -> List[Robot]:
        friendly = [r for rid, r in game.friendly_robots.items() if rid != robot_id]
        enemies = list(game.enemy_robots.values())
        return friendly + enemies

    def _segment_collides(self, game: Game, robot: Robot, start: Vector2D, end: Vector2D) -> bool:
        seg_vec = end - start
        our_velocity_vector = seg_vec / self._simulate_timestep
        our_safety_rect = self._create_safety_rectangle(robot.p, our_velocity_vector)

        for obstacle in self._get_obstacles(game, robot.id):
            obstacle_safety_rect = self._create_safety_rectangle(obstacle.p, obstacle.v)
            if our_safety_rect.intersects(obstacle_safety_rect):
                return True

        return False

    def _random_candidate_angles(self, base_angle: float, count: int) -> List[float]:
        if count <= 0:
            return []
        bin_width = 2 * math.pi / count
        return [
            normalise_heading(base_angle + random.uniform(-math.pi + i * bin_width, -math.pi + (i + 1) * bin_width))
            for i in range(count)
        ]

    def _evaluate_segment(
        self,
        game: Game,
        robot: Robot,
        start: Vector2D,
        end: Vector2D,
        target: Vector2D,
    ) -> float:
        """Evaluate a candidate motion segment; higher score is better."""
        seg_vec = end - start

        start_dist = target.distance_to(start)
        end_dist = target.distance_to(end)
        target_progress_factor = start_dist - end_dist

        our_velocity_vector = seg_vec / self._simulate_timestep
        resulting_speed = our_velocity_vector.mag()
        our_position = robot.p

        # Create our safety rectangle based on candidate velocity
        our_safety_rect = self._create_safety_rectangle(our_position, our_velocity_vector)

        obstacle_factor = 0.0
        for obstacle in self._get_obstacles(game, robot.id):
            # Create safety rectangle for obstacle
            obstacle_safety_rect = self._create_safety_rectangle(obstacle.p, obstacle.v)

            # Check for intersection
            if our_safety_rect.intersects(obstacle_safety_rect):
                # Calculate intersection ratio (0.0 to 1.0)
                intersection_ratio = our_safety_rect.intersection_ratio(obstacle_safety_rect)

                # Exponential penalty: ranges from ~0 to 10
                # Small overlap (~0.1): penalty ≈ 0.1
                # Medium overlap (~0.5): penalty ≈ 2.5
                # Large overlap (~0.9): penalty ≈ 8.1
                # Complete overlap (1.0): penalty = 10.0
                penalty = 10.0 * (intersection_ratio**2)

                obstacle_factor = max(obstacle_factor, penalty)

                # This should be commented out in real execution to speed up further
                self._draw_collision_box(obstacle_safety_rect, "red" if intersection_ratio > 0.5 else "orange", 2)

            else:
                # Calculate distance between rectangles
                dist = our_safety_rect.distance_to(obstacle_safety_rect)

                # Apply penalty based on proximity
                # Close but not intersecting still gets some penalty
                if dist < 0.5:  # Within 0.5m
                    # Exponential decay: closer = higher penalty
                    # dist=0.0m: penalty ≈ 1.0
                    # dist=0.2m: penalty ≈ 0.45
                    # dist=0.5m: penalty ≈ 0.08
                    penalty = 1.0 * math.exp(-4.0 * dist)
                    obstacle_factor = max(obstacle_factor, penalty)

                # This should be commented out in real execution to speed up further
                self._draw_collision_box(obstacle_safety_rect, "green", 1)

        # Use configured weights
        score = (
            self._w_goal * target_progress_factor + self._w_speed * resulting_speed - self._w_obstacle * obstacle_factor
        )
        return score

    def _draw_collision_box(self, collision_box: OrientedRectangle, color: str, width: int):
        if self.env is not None and self._config.show_debug_rectangles:
            # Visualize safe obstacles in green
            corners = collision_box.get_corners()
            for i in range(4):
                self.env.draw_line([corners[i], corners[(i + 1) % 4]], color=color, width=width)

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

    def _random_candidate_scales(self, distance: float, count: int = 3) -> List[float]:
        min_scale = 0.05
        # linear scaling: distance=2 → 1.0, distance=0 → min_scale
        max_scale = min_scale + (1.0 - min_scale) * min(distance / 2.0, 1.0)
        if count <= 0:
            return []
        if max_scale - min_scale <= 1e-9:
            return [min_scale]

        bin_width = (max_scale - min_scale) / count
        samples = [random.uniform(min_scale + i * bin_width, min_scale + (i + 1) * bin_width) for i in range(count)]
        return sorted(samples, reverse=True)

    def _segment_overshoots_target(self, start: Vector2D, end: Vector2D, target: Vector2D) -> bool:
        """Return True when the candidate segment would pass through the target."""
        segment = end - start
        seg_len_sq = segment.dot(segment)
        if seg_len_sq <= 1e-9:
            return False

        to_target = target - start
        dist_start = to_target.mag()
        dist_end = end.distance_to(target)
        overshoot_tolerance = getattr(self._config, "overshoot_tolerance", self._config.target_tolerance * 3.0)
        overshoot_tolerance = max(overshoot_tolerance, self._config.target_tolerance)

        if dist_start <= overshoot_tolerance and dist_end > dist_start:
            return True

        projection = to_target.dot(segment)
        if 0.0 < projection < seg_len_sq:
            distance = point_segment_distance(target, start, end)
            if distance <= overshoot_tolerance:
                return True

        if dist_start > 1e-9:
            direction_to_target = to_target / dist_start
            end_offset = end - start
            along = end_offset.dot(direction_to_target)
            if along > dist_start:
                lateral = end_offset - direction_to_target * along
                if lateral.mag() <= overshoot_tolerance:
                    return True

        return False
