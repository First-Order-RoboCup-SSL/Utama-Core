"""Strategy for random movement within bounded area."""

import random
import time
from typing import Dict, Optional, Tuple

import py_trees

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


class RandomMovementBehaviour(AbstractBehaviour):
    """
    Behaviour that makes a robot move to random targets within bounds.

    Args:
        robot_id: The robot ID to control
        field_bounds: ((min_x, max_x), (min_y, max_y)) bounds for movement
        min_target_distance: Minimum distance for selecting next target
        endpoint_tolerance: Distance to consider target reached
        speed_range: (min_speed, max_speed) for random speed selection
    """

    def __init__(
        self,
        robot_id: int,
        field_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        min_target_distance: float,
        endpoint_tolerance: float,
        speed_range: Tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__(name=f"RandomMovement_{robot_id}")
        self.robot_id = robot_id
        self.field_bounds = field_bounds
        self.min_target_distance = min_target_distance
        self.endpoint_tolerance = endpoint_tolerance
        self.speed_range = speed_range

        self.current_target = None
        self.current_speed = None
        self.targets_reached = 0

    def _generate_random_target(self, current_pos: Vector2D) -> Vector2D:
        """Generate a random target position within bounds, min distance away from current position."""
        (min_x, max_x), (min_y, max_y) = self.field_bounds

        max_attempts = 50
        for _ in range(max_attempts):
            x = random.uniform(min_x + 0.3, max_x - 0.3)
            y = random.uniform(min_y + 0.3, max_y - 0.3)
            target = Vector2D(x, y)

            # Check if target is far enough from current position
            distance = current_pos.distance_to(target)
            if distance >= self.min_target_distance:
                return target

        # Fallback: just return a random position even if too close
        x = random.uniform(min_x + 0.3, max_x - 0.3)
        y = random.uniform(min_y + 0.3, max_y - 0.3)
        return Vector2D(x, y)

    def initialise(self):
        """Initialize with a random target and speed."""
        # Will set target on first update when we have robot position
        self.current_target = None
        self.current_speed = random.uniform(*self.speed_range)

    def update(self) -> py_trees.common.Status:
        """Command robot to move to random targets."""
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env

        if not game.friendly_robots or self.robot_id not in game.friendly_robots:
            return py_trees.common.Status.RUNNING

        robot = game.friendly_robots[self.robot_id]
        robot_pos = Vector2D(robot.p.x, robot.p.y)

        # Generate initial target if needed
        if self.current_target is None:
            self.current_target = self._generate_random_target(robot_pos)
            self.current_speed = random.uniform(*self.speed_range)

        # Check if target reached
        distance_to_target = robot_pos.distance_to(self.current_target)
        if distance_to_target <= self.endpoint_tolerance:
            # Target reached! Generate new target
            self.targets_reached += 1

            # Notify test manager if available
            if hasattr(self.blackboard, "test_manager") and self.blackboard.test_manager:
                self.blackboard.test_manager.update_target_reached(self.robot_id)

            # Generate new target and speed
            self.current_target = self._generate_random_target(robot_pos)
            self.current_speed = random.uniform(*self.speed_range)

        # Visualize target
        if rsim_env:
            rsim_env.draw_point(self.current_target.x, self.current_target.y, color="green")
            # Draw a line to show path
            rsim_env.draw_line(
                [(robot_pos.x, robot_pos.y), (self.current_target.x, self.current_target.y)],
                color="blue",
                width=1,
            )

        # Generate movement command
        cmd = move(
            game,
            self.blackboard.motion_controller,
            self.robot_id,
            self.current_target,
            0.0,  # Face forward
        )

        self.blackboard.cmd_map[self.robot_id] = cmd
        return py_trees.common.Status.RUNNING


class RandomMovementStrategy(AbstractStrategy):
    """
    Strategy that controls multiple robots moving randomly within bounds.

    Args:
        n_robots: Number of robots to control
        field_bounds: ((min_x, max_x), (min_y, max_y)) bounds for movement
        min_target_distance: Minimum distance for selecting next target
        endpoint_tolerance: Distance to consider target reached
        speed_range: (min_speed, max_speed) for random speed selection
    """

    def __init__(
        self,
        n_robots: int,
        field_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        min_target_distance: float,
        endpoint_tolerance: float,
        speed_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.n_robots = n_robots
        self.field_bounds = field_bounds
        self.min_target_distance = min_target_distance
        self.endpoint_tolerance = endpoint_tolerance
        self.speed_range = speed_range
        self.test_manager = None
        super().__init__()

    def set_test_manager(self, test_manager):
        """Set test manager to track targets reached."""
        self.test_manager = test_manager

    def setup_behaviour_tree(self, is_opp_strat: bool = False):
        """Override to set test_manager in blackboard."""
        super().setup_behaviour_tree(is_opp_strat)
        if self.test_manager:
            self.blackboard.test_manager = self.test_manager

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """Requires number of friendly robots to match."""
        return n_runtime_friendly >= self.n_robots

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        """Return the movement bounds."""
        (min_x, max_x), (min_y, max_y) = self.field_bounds

        padding = 0.5
        return FieldBounds(
            top_left=(min_x - padding, max_y + padding),
            bottom_right=(max_x + padding, min_y - padding),
        )

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Create parallel behaviour tree with all robot random movement behaviours."""
        if self.n_robots == 1:
            # Single robot
            return RandomMovementBehaviour(
                robot_id=0,
                field_bounds=self.field_bounds,
                min_target_distance=self.min_target_distance,
                endpoint_tolerance=self.endpoint_tolerance,
                speed_range=self.speed_range,
            )

        # Multiple robots - create parallel behaviours
        behaviours = []
        for robot_id in range(self.n_robots):
            behaviour = RandomMovementBehaviour(
                robot_id=robot_id,
                field_bounds=self.field_bounds,
                min_target_distance=self.min_target_distance,
                endpoint_tolerance=self.endpoint_tolerance,
                speed_range=self.speed_range,
            )
            behaviours.append(behaviour)

        # Run all robot behaviours in parallel
        return py_trees.composites.Parallel(
            name="RandomMovement",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=behaviours,
        )
