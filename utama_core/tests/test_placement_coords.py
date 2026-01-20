"""Integration tests for RobotPlacementStrategy using AbstractTestManager."""

import math

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
    """Test manager for the RobotPlacement strategy."""

    n_episodes = 1  # Single episode - just verify robot moves to expected region

    def __init__(self, expected_center: tuple[float, float]):
        super().__init__()
        self.expected_center = expected_center
        self.frames_in_region = 0
        self.required_frames = 30  # Robot must be in region for 30 frames

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """Reset position of robots and ball for the next strategy test."""
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

        # Start robot at origin - it should move to expected_center region
        sim_controller.teleport_robot(
            game.my_team_is_yellow,
            self.my_strategy.robot_id,
            0,
            0,
        )
        sim_controller.teleport_ball(2, 2)  # Ball out of the way

    def eval_status(self, game: Game) -> TestingStatus:
        """Evaluate if robot is oscillating around the expected center."""
        robot = game.friendly_robots.get(self.my_strategy.robot_id)
        if not robot:
            return TestingStatus.IN_PROGRESS

        # Check if robot is within 1m of expected center (oscillation region)
        dist_to_center = math.dist((robot.p.x, robot.p.y), self.expected_center)

        if dist_to_center < 1.0:
            self.frames_in_region += 1
        else:
            self.frames_in_region = 0

        # Success if robot stays in region for required frames
        if self.frames_in_region >= self.required_frames:
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_robot_placement_custom_bounds(
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = False,
    robot_id: int = 0,
    headless: bool = True,
    mode: str = "rsim",
):
    """Test that RobotPlacementStrategy places robot at custom bounds center."""
    # Custom bounds centered at (2, 1)
    custom_bounds = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
    expected_center = custom_bounds.center  # (2.0, 1.0)

    runner = StrategyRunner(
        strategy=RobotPlacementStrategy(robot_id=robot_id, field_bounds=custom_bounds),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=1,
        exp_enemy=0,
    )
    test_manager = RobotPlacementTestManager(expected_center=expected_center)
    test = runner.run_test(test_manager=test_manager, episode_timeout=10, rsim_headless=headless)
    assert test, f"Robot did not reach expected center region {expected_center}"


def test_robot_placement_default_bounds(
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = False,
    robot_id: int = 0,
    headless: bool = True,
    mode: str = "rsim",
):
    """Test that RobotPlacementStrategy uses full field center by default."""
    # Default bounds should center at (0, 0)
    expected_center = (0.0, 0.0)

    runner = StrategyRunner(
        strategy=RobotPlacementStrategy(robot_id=robot_id),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=1,
        exp_enemy=0,
    )
    test_manager = RobotPlacementTestManager(expected_center=expected_center)
    test = runner.run_test(test_manager=test_manager, episode_timeout=10, rsim_headless=headless)
    assert test, f"Robot did not reach expected center region {expected_center}"


if __name__ == "__main__":
    # Run manually for debugging
    test_robot_placement_custom_bounds(headless=False)
