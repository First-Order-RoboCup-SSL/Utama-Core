"""Tests for random movement collision avoidance with multiple robots."""

import os
from dataclasses import dataclass
from typing import Dict

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.run import StrategyRunner
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)
from utama_core.tests.motion_planning.strategies.random_movement_strategy import (
    RandomMovementStrategy,
)

# Fix pygame window position for screen capture
os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"


@dataclass
class RandomMovementScenario:
    """Configuration for random movement collision test."""

    n_robots: int
    field_bounds: tuple[tuple[float, float], tuple[float, float]]  # ((min_x, max_x), (min_y, max_y))
    min_target_distance: float
    required_targets_per_robot: int
    collision_threshold: float = ROBOT_RADIUS * 2.0
    endpoint_tolerance: float = 0.25


class RandomMovementTestManager(AbstractTestManager):
    """Test manager for random movement with collision detection."""

    def __init__(self, scenario: RandomMovementScenario):
        super().__init__()
        self.scenario = scenario
        self.n_episodes = 1
        self.collision_detected = False
        self.min_distance = float("inf")
        self.collision_count = 0
        self.targets_reached_count: Dict[int, int] = {}  # Track how many targets each robot has reached

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """Reset field with robots in random starting positions within bounds."""
        import random

        (min_x, max_x), (min_y, max_y) = self.scenario.field_bounds

        # Place robots at random positions within bounds
        for i in range(self.scenario.n_robots):
            x = random.uniform(min_x + 0.5, max_x - 0.5)
            y = random.uniform(min_y + 0.5, max_y - 0.5)
            sim_controller.teleport_robot(
                game.my_team_is_yellow,
                i,
                x,
                y,
                0.0,
            )
            self.targets_reached_count[i] = 0

        # Teleport remaining robots far away
        for i in range(self.scenario.n_robots, 6):
            sim_controller.teleport_robot(
                game.my_team_is_yellow,
                i,
                -10.0,
                -10.0,
                0.0,
            )

        # Remove enemy robots
        for i in range(6):
            sim_controller.teleport_robot(
                not game.my_team_is_yellow,
                i,
                -10.0,
                -10.0,
                0.0,
            )

        # Place ball out of the way
        sim_controller.teleport_ball(-10.0, -10.0)

        self._reset_metrics()

    def _reset_metrics(self):
        """Reset tracking metrics for new episode."""
        self.collision_detected = False
        self.min_distance = float("inf")
        self.collision_count = 0

    def eval_status(self, game: Game):
        """Evaluate collision status and target achievement."""
        # Check collisions between all pairs of friendly robots
        for robot_id, robot in game.friendly_robots.items():
            robot_pos = Vector2D(robot.p.x, robot.p.y)

            # Check collisions with other friendly robots
            for other_id, other_robot in game.friendly_robots.items():
                if robot_id >= other_id:
                    continue
                other_pos = Vector2D(other_robot.p.x, other_robot.p.y)
                distance = robot_pos.distance_to(other_pos)
                self.min_distance = min(self.min_distance, distance)

                if distance < self.scenario.collision_threshold:
                    self.collision_detected = True
                    self.collision_count += 1
                    return TestingStatus.FAILURE

        # Check if all robots have reached required number of targets
        # This is tracked by the strategy itself
        all_completed = all(
            count >= self.scenario.required_targets_per_robot for count in self.targets_reached_count.values()
        )
        if all_completed:
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS

    def get_n_episodes(self):
        return self.n_episodes

    def update_target_reached(self, robot_id: int):
        """Called by strategy when a robot reaches a target."""
        if robot_id in self.targets_reached_count:
            self.targets_reached_count[robot_id] += 1


def test_random_movement_same_team(
    headless: bool,
    mode: str = "rsim",
):
    """
    Test where 2 robots from the same team move randomly within half court.

    The robots should:
    1. Start at random positions within their half court
    2. Navigate to random targets with minimum distance requirement
    3. Each robot reaches at least 3 different targets
    4. Avoid collisions with all teammates throughout the movement
    """
    my_team_is_yellow = True
    my_team_is_right = False  # Yellow on left half

    # Define half court bounds for left side (Yellow team)
    # Standard SSL field is ~9m x 6m, so half court is ~4.5m x 6m
    # Using slightly smaller bounds for safety: -4m to 0m in x, -2.5m to 2.5m in y
    field_bounds = (
        (-4.0, -0.5),
        (-2.0, 2.0),
    )  # ((min_x, max_x), (min_y, max_y))

    # Max is 6 robots
    n_robots = 2

    scenario = RandomMovementScenario(
        n_robots=n_robots,
        field_bounds=field_bounds,
        min_target_distance=1.0,  # Minimum distance for next target
        required_targets_per_robot=3,  # Each robot must reach 3 targets
        endpoint_tolerance=0.3,
    )

    # Create random movement strategy
    strategy = RandomMovementStrategy(
        n_robots=n_robots,
        field_bounds=field_bounds,
        min_target_distance=scenario.min_target_distance,
        endpoint_tolerance=scenario.endpoint_tolerance,
        test_manager=RandomMovementTestManager(scenario),
        speed_range=(0.5, 1.0),  # Random speed between 0.5 and 2.0 m/s
    )

    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=n_robots,
        exp_enemy=0,
    )

    test_passed = runner.run_test(
        test_manager=strategy.test_manager,
        episode_timeout=60.0,  # 60 seconds to complete random movements
        rsim_headless=headless,
    )

    # Assertions
    assert test_passed, "Random movement test failed to complete"

    # Check that all robots reached required targets
    for robot_id in range(n_robots):
        assert strategy.test_manager.targets_reached_count[robot_id] >= scenario.required_targets_per_robot, (
            f"Robot {robot_id} only reached {strategy.test_manager.targets_reached_count[robot_id]} targets "
            f"(required: {scenario.required_targets_per_robot})"
        )

    assert not strategy.test_manager.collision_detected, (
        f"Robots collided {strategy.test_manager.collision_count} time(s) during random movement! "
        f"Minimum distance: {strategy.test_manager.min_distance:.3f}m "
        f"(threshold: {scenario.collision_threshold:.3f}m)"
    )
    assert strategy.test_manager.min_distance >= scenario.collision_threshold, (
        f"Robots got too close: {strategy.test_manager.min_distance:.3f}m "
        f"(minimum safe distance: {scenario.collision_threshold:.3f}m)"
    )
