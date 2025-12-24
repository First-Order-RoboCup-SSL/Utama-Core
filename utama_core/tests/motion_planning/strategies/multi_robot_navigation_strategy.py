"""Strategy for controlling multiple robots, each with their own target position."""

from typing import Dict, Optional

import py_trees

from utama_core.entities.game.field import FieldBounds
from utama_core.strategy.common.abstract_strategy import AbstractStrategy

from .simple_navigation_strategy import NavigateToTarget


class MultiRobotNavigationStrategy(AbstractStrategy):
    """
    Strategy that controls multiple robots, each with their own target position.

    Args:
        robot_targets: Dictionary mapping robot_id to target position (x, y)
        target_orientation: Optional target orientation for all robots (default: 0)
    """

    def __init__(
        self,
        robot_targets: Dict[int, tuple[float, float]],
        target_orientation: float = 0.0,
    ):
        self.robot_targets = robot_targets
        self.target_orientation = target_orientation
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """Requires number of friendly robots to match target count."""
        return n_runtime_friendly >= len(self.robot_targets)

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        """Calculate bounding box for all robot targets."""
        if not self.robot_targets:
            return None

        all_x = [pos[0] for pos in self.robot_targets.values()]
        all_y = [pos[1] for pos in self.robot_targets.values()]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Add padding
        padding = 1.0
        return FieldBounds(
            top_left=(min_x - padding, max_y + padding),
            bottom_right=(max_x + padding, min_y - padding),
        )

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Create parallel behaviour tree with all robot navigation behaviours."""
        if len(self.robot_targets) == 1:
            # Single robot - just return one behaviour
            robot_id, target = list(self.robot_targets.items())[0]
            return NavigateToTarget(
                robot_id=robot_id,
                target_position=target,
                target_orientation=self.target_orientation,
            )

        # Multiple robots - create parallel behaviours
        behaviours = []
        for robot_id, target in self.robot_targets.items():
            behaviour = NavigateToTarget(
                robot_id=robot_id,
                target_position=target,
                target_orientation=self.target_orientation,
            )
            behaviours.append(behaviour)

        # Run all robot navigation behaviours in parallel
        return py_trees.composites.Parallel(
            name="MultiRobotNavigation",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=behaviours,
        )
