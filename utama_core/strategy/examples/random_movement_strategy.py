"""Strategy for random movement within bounded area."""

from __future__ import annotations

import random
from typing import Optional

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
        robot_id: The robot ID to control.
        field_bounds: Movement bounds.
        endpoint_tolerance: Distance to consider target reached.
        min_target_distance: Minimum distance from current position when selecting next target.
        seed: Random seed for deterministic behaviour.
        on_target_reached: Optional callback invoked with robot_id when a target is reached.
    """

    def __init__(
        self,
        robot_id: int,
        field_bounds: FieldBounds,
        endpoint_tolerance: float,
        min_target_distance: float = 0.0,
        seed: Optional[int] = None,
        on_target_reached: Optional[callable] = None,
    ):
        super().__init__(name=f"RandomMovement_{robot_id}")
        self.robot_id = robot_id
        self.field_bounds = field_bounds
        self.endpoint_tolerance = endpoint_tolerance
        self.min_target_distance = min_target_distance
        self.on_target_reached = on_target_reached

        self.current_target: Optional[Vector2D] = None
        self.rng = random.Random(seed)

        # Pre-compute bounds for sampling
        self._min_x = min(field_bounds.top_left[0], field_bounds.bottom_right[0])
        self._max_x = max(field_bounds.top_left[0], field_bounds.bottom_right[0])
        self._min_y = min(field_bounds.top_left[1], field_bounds.bottom_right[1])
        self._max_y = max(field_bounds.top_left[1], field_bounds.bottom_right[1])

    def _generate_random_target(self, current_pos: Vector2D) -> Vector2D:
        """Generate a random target within bounds, at least min_target_distance from current pos."""
        MAX_ATTEMPTS = 50
        PADDING = 0.3

        for _ in range(MAX_ATTEMPTS):
            x = self.rng.uniform(self._min_x + PADDING, self._max_x - PADDING)
            y = self.rng.uniform(self._min_y + PADDING, self._max_y - PADDING)
            target = Vector2D(x, y)

            if current_pos.distance_to(target) >= self.min_target_distance:
                return target

        # Fallback: return a random position even if too close
        x = self.rng.uniform(self._min_x + PADDING, self._max_x - PADDING)
        y = self.rng.uniform(self._min_y + PADDING, self._max_y - PADDING)
        return Vector2D(x, y)

    def initialise(self):
        self.current_target = None

    def update(self) -> py_trees.common.Status:
        """Command robot to move to random targets."""
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env

        if not game.friendly_robots or self.robot_id not in game.friendly_robots:
            return py_trees.common.Status.RUNNING

        robot = game.friendly_robots[self.robot_id]
        robot_pos = Vector2D(robot.p.x, robot.p.y)

        if self.current_target is None:
            self.current_target = self._generate_random_target(robot_pos)

        if robot_pos.distance_to(self.current_target) <= self.endpoint_tolerance:
            if self.on_target_reached is not None:
                self.on_target_reached(self.robot_id)
            self.current_target = self._generate_random_target(robot_pos)

        if rsim_env:
            rsim_env.draw_point(self.current_target.x, self.current_target.y, color="green")
            rsim_env.draw_line(
                [(robot_pos.x, robot_pos.y), (self.current_target.x, self.current_target.y)],
                color="blue",
                width=1,
            )

        cmd = move(
            game,
            self.blackboard.motion_controller,
            self.robot_id,
            self.current_target,
            0.0,
        )

        self.blackboard.cmd_map[self.robot_id] = cmd
        return py_trees.common.Status.RUNNING


class RandomMovementStrategy(AbstractStrategy):
    """
    Strategy that controls multiple robots moving randomly within bounds.

    Args:
        n_robots: Number of robots to control.
        field_bounds: Movement bounds.
        endpoint_tolerance: Distance to consider target reached.
        min_target_distance: Minimum distance from current position when selecting next target.
        seed: Base seed for deterministic behaviour. Each robot gets ``seed + robot_id``.
        on_target_reached: Optional callback invoked with robot_id when a target is reached.
    """

    exp_ball = False

    def __init__(
        self,
        n_robots: int,
        field_bounds: FieldBounds,
        endpoint_tolerance: float,
        min_target_distance: float = 0.0,
        seed: Optional[int] = None,
        on_target_reached: Optional[callable] = None,
    ):
        self.n_robots = n_robots
        self.field_bounds = field_bounds
        self.endpoint_tolerance = endpoint_tolerance
        self.min_target_distance = min_target_distance
        self.seed = seed
        self.on_target_reached = on_target_reached
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return n_runtime_friendly >= self.n_robots

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        return self.field_bounds

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Create parallel behaviour tree with one RandomMovementBehaviour per robot."""

        def make_behaviour(robot_id: int) -> RandomMovementBehaviour:
            return RandomMovementBehaviour(
                robot_id=robot_id,
                field_bounds=self.field_bounds,
                endpoint_tolerance=self.endpoint_tolerance,
                min_target_distance=self.min_target_distance,
                seed=None if self.seed is None else self.seed + robot_id,
                on_target_reached=self.on_target_reached,
            )

        if self.n_robots == 1:
            return make_behaviour(0)

        return py_trees.composites.Parallel(
            name="RandomMovement",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[make_behaviour(robot_id) for robot_id in range(self.n_robots)],
        )
