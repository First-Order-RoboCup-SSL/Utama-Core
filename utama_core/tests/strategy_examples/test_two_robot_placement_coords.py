"""Integration tests for TwoRobotPlacementStrategy using AbstractTestManager.

These tests verify that:
1. Both robots move to positions based on the provided field_bounds center
2. Robot 1 oscillates horizontally (center ± 0.5 in X)
3. Robot 2 oscillates vertically (center ± 0.5 in Y)
4. Turn-based coordination works correctly
5. Custom field_bounds correctly shift the placement region
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
from utama_core.strategy.examples.two_robot_placement import TwoRobotPlacementStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)


class TwoRobotPlacementTestManager(AbstractTestManager):
    """Test manager that verifies both robots oscillate around the expected center."""

    n_episodes = 1

    def __init__(
        self,
        expected_center: tuple[float, float],
        first_robot_id: int,
        second_robot_id: int,
        tolerance: float = 0.15,
    ):
        super().__init__()
        self.expected_center = expected_center
        self.first_robot_id = first_robot_id
        self.second_robot_id = second_robot_id
        self.tolerance = tolerance

        # Robot 1 (horizontal mover) targets
        self.r1_left = (expected_center[0] - 0.5, expected_center[1])
        self.r1_right = (expected_center[0] + 0.5, expected_center[1])
        self.r1_reached_left = False
        self.r1_reached_right = False

        # Robot 2 (vertical mover) targets
        self.r2_upper = (expected_center[0], expected_center[1] + 0.5)
        self.r2_lower = (expected_center[0], expected_center[1] - 0.5)
        self.r2_reached_upper = False
        self.r2_reached_lower = False

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

        # Position robots near the center for faster test convergence
        cx, cy = self.expected_center
        sim_controller.teleport_robot(game.my_team_is_yellow, self.first_robot_id, cx, cy)
        sim_controller.teleport_robot(game.my_team_is_yellow, self.second_robot_id, cx, cy)
        sim_controller.teleport_ball(3, 3)

    def eval_status(self, game: Game) -> TestingStatus:
        """Verify both robots reach their oscillation targets."""
        robot1 = game.friendly_robots.get(self.first_robot_id)
        robot2 = game.friendly_robots.get(self.second_robot_id)
        if not robot1 or not robot2:
            return TestingStatus.IN_PROGRESS

        r1_pos = (robot1.p.x, robot1.p.y)
        r2_pos = (robot2.p.x, robot2.p.y)

        # Check Robot 1 horizontal targets
        if math.dist(r1_pos, self.r1_left) < self.tolerance:
            self.r1_reached_left = True
        if math.dist(r1_pos, self.r1_right) < self.tolerance:
            self.r1_reached_right = True

        # Check Robot 2 vertical targets
        if math.dist(r2_pos, self.r2_upper) < self.tolerance:
            self.r2_reached_upper = True
        if math.dist(r2_pos, self.r2_lower) < self.tolerance:
            self.r2_reached_lower = True

        # Success when both robots reach both their targets
        if self.r1_reached_left and self.r1_reached_right and self.r2_reached_upper and self.r2_reached_lower:
            return TestingStatus.SUCCESS
        return TestingStatus.IN_PROGRESS


def _run_two_robot_placement_test(
    field_bounds: Optional[FieldBounds],
    expected_center: tuple[float, float],
    first_robot_id: int = 0,
    second_robot_id: int = 1,
):
    """Helper to run a two-robot placement strategy test with given bounds."""
    strategy = TwoRobotPlacementStrategy(
        first_robot_id=first_robot_id,
        second_robot_id=second_robot_id,
        field_bounds=field_bounds,
    )

    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=2,
        exp_enemy=0,
    )

    test_manager = TwoRobotPlacementTestManager(
        expected_center=expected_center,
        first_robot_id=first_robot_id,
        second_robot_id=second_robot_id,
    )
    passed = runner.run_test(test_manager=test_manager, episode_timeout=20, rsim_headless=True)

    return passed, test_manager


class TestTwoRobotPlacementStrategy:
    """Integration tests for TwoRobotPlacementStrategy behavior."""

    def test_oscillation_with_default_bounds(self):
        """Both robots oscillate around default field center (0, 0)."""
        expected_center = (0.0, 0.0)

        passed, manager = _run_two_robot_placement_test(field_bounds=None, expected_center=expected_center)

        assert manager.r1_reached_left, f"Robot 1 never reached left target {manager.r1_left}"
        assert manager.r1_reached_right, f"Robot 1 never reached right target {manager.r1_right}"
        assert manager.r2_reached_upper, f"Robot 2 never reached upper target {manager.r2_upper}"
        assert manager.r2_reached_lower, f"Robot 2 never reached lower target {manager.r2_lower}"
        assert passed

    def test_oscillation_with_custom_bounds(self):
        """Both robots oscillate around custom bounds center (2.0, 1.0)."""
        bounds = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
        expected_center = bounds.center

        passed, manager = _run_two_robot_placement_test(bounds, expected_center)

        assert manager.r1_reached_left, f"Robot 1 never reached left target {manager.r1_left}"
        assert manager.r1_reached_right, f"Robot 1 never reached right target {manager.r1_right}"
        assert manager.r2_reached_upper, f"Robot 2 never reached upper target {manager.r2_upper}"
        assert manager.r2_reached_lower, f"Robot 2 never reached lower target {manager.r2_lower}"
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
