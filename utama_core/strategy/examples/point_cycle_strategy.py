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
        field_bounds: FieldBounds,
        endpoint_tolerance: float
    ):
        super().__init__(name=f"RandomPoint_{robot_id}")
        self.robot_id = robot_id
        self.field_bounds = field_bounds
        self.endpoint_tolerance = endpoint_tolerance

        self.current_target = None
        self.points = PointCycle()


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
    Strategy that controls multiple robots moving between a set of points within bounds.

    Args:
        n_robots: Number of robots to control
        field_bounds: ((min_x, max_x), (min_y, max_y)) bounds for movement
        endpoint_tolerance: Distance to consider target reached
    """

    def __init__(
        self,
        n_robots: int,
        field_bounds: FieldBounds,
        endpoint_tolerance: float,
    ):
        self.n_robots = n_robots
        self.field_bounds = field_bounds
        self.endpoint_tolerance = endpoint_tolerance
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
            # Single robot
            return PointCycleBehaviour(
                robot_id=0,
                field_bounds=self.field_bounds,
                endpoint_tolerance=self.endpoint_tolerance,
            )

        # Multiple robots - create parallel behaviours
        behaviours = []
        for robot_id in range(self.n_robots):
            behaviour = PointCycleBehaviour(
                robot_id=robot_id,
                field_bounds=self.field_bounds,
                endpoint_tolerance=self.endpoint_tolerance,
            )
            behaviours.append(behaviour)

        # Run all robot behaviours in parallel
        return py_trees.composites.Parallel(
            name="RandomPoint",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=behaviours,
        )


class PointCycle:
    def __init__(self):
        self.pointqueue = deque([
            Vector2D(x=-0.8788985666614408, y=1.833059669991835),
            Vector2D(x=1.9214130018502678, y=-1.8036117772036704),
            Vector2D(x=-2.9433944382347486, y=0.6953808069339509),
            Vector2D(x=-3.4181903822441044, y=-2.56652611346848),
            Vector2D(x=-2.5460022017253, y=0.7045762098821444),
            Vector2D(x=-4.155487299825273, y=-1.2398233473829672),
            Vector2D(x=3.5298532254117605, y=-1.9554006897411762),
            Vector2D(x=-1.2978745719981406, y=0.5846383006608473),
            Vector2D(x=-3.5955893359815656, y=-2.6009526656219326),
            Vector2D(x=-2.7151516250314636, y=0.9617686362412114)
        ])
    
    @property
    def next_point(self):
        next = self.pointqueue.pop()
        self.pointqueue.appendleft(next)
        return next