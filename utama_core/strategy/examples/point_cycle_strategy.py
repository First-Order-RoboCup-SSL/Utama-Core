"""Strategy for random movement within bounded area."""

from __future__ import annotations

import random
from collections import deque
from typing import Optional, Tuple

import py_trees

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


class PointCycleBehaviour(AbstractBehaviour):
    """
    Behaviour that makes a robot move to randomly sampled targets within bounds.

    Args:
        robot_id (int): The robot ID to control.
        field_bounds (FieldBounds): ((min_x, max_x), (min_y, max_y)) bounds for movement.
        endpoint_tolerance (float): Distance to consider target reached.
        seed (Optional[int]): Seed for deterministic random sampling.
    """

    def __init__(
        self,
        robot_id: int,
        field_bounds: FieldBounds,
        endpoint_tolerance: float,
        seed: Optional[int] = None,
    ):
        super().__init__(name=f"RandomPoint_{robot_id}")
        self.robot_id = robot_id
        self.field_bounds = field_bounds
        self.endpoint_tolerance = endpoint_tolerance

        self.current_target = None
        self.points = RandomPointSampler(field_bounds, seed=seed)

    def initialise(self):
        """Initialize with a random target and speed."""
        # Will set target on first update when we have robot position
        pass

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
            self.current_target = self.points.next_point

        # Check if target reached
        distance_to_target = robot_pos.distance_to(self.current_target)
        if distance_to_target <= self.endpoint_tolerance:
            # Generate new target and speed
            self.current_target = self.points.next_point

        # Visualize target
        if rsim_env:
            rsim_env.draw_point(self.current_target.x, self.current_target.y, color="green")
            # Draw a line to show path
            rsim_env.draw_line(
                [
                    (robot_pos.x, robot_pos.y),
                    (self.current_target.x, self.current_target.y),
                ],
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


class PointCycleStrategy(AbstractStrategy):
    """
    Strategy that instantiates one PointCycleBehaviour per friendly robot
    and executes them in parallel within specified field bounds.

    Args:
        n_robots (int): Number of robots to control.
        field_bounds (FieldBounds): Movement bounds.
        endpoint_tolerance (float): Distance to consider target reached.
        seed (Optional[int]): Base seed for deterministic behaviour.
    """

    exp_ball = False  # This behaviour does not require the ball to be present

    def __init__(
        self,
        n_robots: int,
        field_bounds: FieldBounds,
        endpoint_tolerance: float,
        seed: Optional[int] = None,
    ):
        self.n_robots = n_robots
        self.field_bounds = field_bounds
        self.endpoint_tolerance = endpoint_tolerance
        self.seed = seed
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """Requires number of friendly robots to match."""
        return n_runtime_friendly >= self.n_robots

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        """Return the movement bounds."""
        return self.field_bounds

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Create parallel behaviour tree with all robot random movement behaviours."""
        if self.n_robots == 1:
            return PointCycleBehaviour(
                robot_id=0,
                field_bounds=self.field_bounds,
                endpoint_tolerance=self.endpoint_tolerance,
                seed=self.seed,
            )

        behaviours = []
        for robot_id in range(self.n_robots):
            behaviour = PointCycleBehaviour(
                robot_id=robot_id,
                field_bounds=self.field_bounds,
                endpoint_tolerance=self.endpoint_tolerance,
                seed=None if self.seed is None else self.seed + robot_id,
            )
            behaviours.append(behaviour)

        return py_trees.composites.Parallel(
            name="RandomPoint",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=behaviours,
        )


class RandomPointSampler:
    """
    Uniform random point sampler within rectangular field bounds.

    Args:
        field_bounds (FieldBounds): ((min_x, max_x), (min_y, max_y))
        seed (Optional[int]): Random seed for deterministic sampling.
    """

    def __init__(self, field_bounds: FieldBounds, seed: int = 42):
        self.field_bounds = field_bounds
        self.rng = random.Random(seed)

    @property
    def next_point(self) -> Vector2D:
        min_x = self.field_bounds.bottom_right[0]
        max_x = self.field_bounds.top_left[0]
        min_y = self.field_bounds.bottom_right[1]
        max_y = self.field_bounds.top_left[1]

        x = self.rng.uniform(min_x, max_x)
        y = self.rng.uniform(min_y, max_y)

        return Vector2D(x=x, y=y)
