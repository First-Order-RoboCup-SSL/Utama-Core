"""Simple navigation strategy for testing - moves to a fixed target position."""

from typing import Optional

import py_trees

from utama_core.config.settings import TIMESTEP
from utama_core.entities.game.field import FieldBounds
from utama_core.entities.data.vector import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


class NavigateToTarget(AbstractBehaviour):
    """
    Navigate to a fixed target position.

    Args:
        target_position: The fixed target position (x, y) to navigate to
        target_orientation: Optional target orientation in radians (default: 0)
    """

    def __init__(
        self,
        robot_id: int,
        target_position: tuple[float, float],
        target_orientation: float = 0.0,
    ):
        super().__init__(name=f"NavigateToTarget_{robot_id}")
        self.robot_id = robot_id
        self.target_position = Vector2D(*target_position)
        self.target_orientation = target_orientation

    def update(self) -> py_trees.common.Status:
        """Command robot to move to the fixed target."""
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env

        if not game.friendly_robots or self.robot_id not in game.friendly_robots:
            return py_trees.common.Status.RUNNING

        # Draw target position for visualization
        if rsim_env:
            rsim_env.draw_point(self.target_position.x, self.target_position.y, color="red")

            # Draw predicted position
            robot = game.friendly_robots[self.robot_id]
            v = robot.v
            p = robot.p
            rsim_env.draw_point(p.x + v.x * TIMESTEP * 5, p.y + v.y * TIMESTEP * 5, color="green")

        # Generate movement command
        cmd = move(
            game,
            self.blackboard.motion_controller,
            self.robot_id,
            self.target_position,
            self.target_orientation,
        )

        self.blackboard.cmd_map[self.robot_id] = cmd
        return py_trees.common.Status.RUNNING


class SimpleNavigationStrategy(AbstractStrategy):
    """
    Simple strategy that navigates a robot to a fixed target position.

    Args:
        robot_id: The ID of the robot to control
        target_position: The target position (x, y) to navigate to
        target_orientation: Optional target orientation in radians (default: 0)
    """

    def __init__(
        self,
        robot_id: int,
        target_position: tuple[float, float],
        target_orientation: float = 0.0,
    ):
        self.robot_id = robot_id
        self.target_position = target_position
        self.target_orientation = target_orientation
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """Requires at least 1 friendly robot."""
        return n_runtime_friendly >= 1

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        """Return None to allow full field navigation."""
        return None

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Create behavior tree with fixed target navigation."""
        return NavigateToTarget(
            robot_id=self.robot_id,
            target_position=self.target_position,
            target_orientation=self.target_orientation,
        )
