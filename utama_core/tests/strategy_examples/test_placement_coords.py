"""Integration tests for RobotPlacementStrategy using AbstractTestManager.

These tests verify that:
1. The robot moves to positions based on the provided field_bounds center
2. The robot oscillates between the expected start and end points
3. Custom field_bounds correctly shift the placement region
"""

import math
from typing import Optional

import pytest

from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.mapping_utils import (
    map_friendly_enemy_to_colors,
    map_left_right_to_colors,
)
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.one_robot_placement_strategy import (
    RobotPlacementStrategy,
)
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)


class RobotPlacementTestManager(AbstractTestManager):
    """Test manager that verifies robot oscillates around the expected center."""

    n_episodes = 1

    def __init__(self, expected_center: tuple[float, float], tolerance: float = 0.15):
        super().__init__()
        self.expected_center = expected_center
        self.expected_upper = (expected_center[0], expected_center[1] + 0.5)
        self.expected_lower = (expected_center[0], expected_center[1] - 0.5)
        self.reached_upper = False
        self.reached_lower = False
        self.tolerance = tolerance

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """Reset robot and ball positions for the test."""
        ini_yellow, ini_blue = map_left_right_to_colors(
            game.my_team_is_yellow,
            game.my_team_is_right,
            RIGHT_START_ONE,
            LEFT_START_ONE,
        )

        y_robots, b_robots = map_friendly_enemy_to_colors(
            game.my_team_is_yellow, game.friendly_robots, game.enemy_robots
        )

        for i in b_robots.keys():
            sim_controller.teleport_robot(False, i, ini_blue[i][0], ini_blue[i][1], ini_blue[i][2])
        for j in y_robots.keys():
            sim_controller.teleport_robot(True, j, ini_yellow[j][0], ini_yellow[j][1], ini_yellow[j][2])

        sim_controller.teleport_robot(game.my_team_is_yellow, self.my_strategy.robot_id, 0, 0)
        sim_controller.teleport_ball(3, 3)

    def eval_status(self, game: Game) -> TestingStatus:
        """Verify robot reaches both oscillation targets."""
        robot = game.friendly_robots.get(self.my_strategy.robot_id)
        if not robot:
            return TestingStatus.IN_PROGRESS

        robot_pos = (robot.p.x, robot.p.y)

        if math.dist(robot_pos, self.expected_upper) < self.tolerance:
            self.reached_upper = True
        if math.dist(robot_pos, self.expected_lower) < self.tolerance:
            self.reached_lower = True

        if self.reached_upper and self.reached_lower:
            return TestingStatus.SUCCESS
        return TestingStatus.IN_PROGRESS


def _run_placement_test(field_bounds: Optional[FieldBounds], expected_center: tuple[float, float]):
    """Helper to run a placement strategy test with given bounds."""
    strategy = RobotPlacementStrategy(robot_id=0, field_bounds=field_bounds)

    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
    )

    test_manager = RobotPlacementTestManager(expected_center=expected_center)
    passed = runner.run_test(test_manager=test_manager, episode_timeout=15, rsim_headless=True)

    return passed, test_manager


class TestFieldBoundsCenter:
    """Tests for FieldBounds center calculation."""

    def test_full_field_center_is_origin(self):
        """Full field bounds should have center at (0, 0)."""
        bounds = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
        assert bounds.center == (0.0, 0.0)

    def test_custom_bounds_center(self):
        """Custom bounds should correctly calculate center."""
        bounds = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
        assert bounds.center == (2.0, 1.0)

    def test_custom_differs_from_default(self):
        """Custom and default bounds must have different centers."""
        default = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
        custom = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
        assert default.center != custom.center


class TestRobotPlacementStrategy:
    """Integration tests for RobotPlacementStrategy behavior."""

    def test_oscillation_with_custom_bounds(self):
        """Robot oscillates around custom bounds center (2.0, 1.0)."""
        bounds = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
        expected_center = bounds.center

        passed, manager = _run_placement_test(bounds, expected_center)

        assert manager.reached_upper, f"Never reached upper target {manager.expected_upper}"
        assert manager.reached_lower, f"Never reached lower target {manager.expected_lower}"
        assert passed

    def test_oscillation_with_default_bounds(self):
        """Robot oscillates around default field center (0, 0)."""
        expected_center = (0.0, 0.0)

        passed, manager = _run_placement_test(field_bounds=None, expected_center=expected_center)

        assert manager.reached_upper, f"Never reached upper target {manager.expected_upper}"
        assert manager.reached_lower, f"Never reached lower target {manager.expected_lower}"
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
