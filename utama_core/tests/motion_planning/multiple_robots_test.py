"""Tests for multiple robots collision avoidance scenarios."""

import os
from dataclasses import dataclass

from utama_core.config.physical_constants import MAX_ROBOTS, ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import MultiRobotNavigationStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

# Fix pygame window position for screen capture
os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"


@dataclass
class MultiRobotScenario:
    """Configuration for multi-robot collision test."""

    friendly_positions: list[tuple[float, float]]
    enemy_positions: list[tuple[float, float]]
    friendly_targets: list[tuple[float, float]]
    enemy_targets: list[tuple[float, float]]
    collision_threshold: float = ROBOT_RADIUS * 2.0
    endpoint_tolerance: float = 0.2


class MultiRobotTestManager(AbstractTestManager):
    """Test manager for multiple robots with collision detection."""

    n_episodes = 1

    def __init__(self, scenario: MultiRobotScenario):
        super().__init__()
        self.scenario = scenario
        self.all_reached = False
        self.collision_detected = False
        self.min_distance = float("inf")
        self.collision_count = 0
        self.robots_reached = set()

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """Reset field with all robots at their starting positions."""
        for i, (x, y) in enumerate(self.scenario.friendly_positions):
            sim_controller.teleport_robot(game.my_team_is_yellow, i, x, y, 0.0)

        for i, (x, y) in enumerate(self.scenario.enemy_positions):
            sim_controller.teleport_robot(not game.my_team_is_yellow, i, x, y, 0.0)

        self._reset_metrics()

    def _reset_metrics(self):
        """Reset tracking metrics for new episode."""
        self.all_reached = False
        self.collision_detected = False
        self.min_distance = float("inf")
        self.collision_count = 0
        self.robots_reached = set()

    def eval_status(self, game: Game):
        """Evaluate collision status and goal achievement for all robots."""
        for robot_id, robot in game.friendly_robots.items():
            robot_pos = Vector2D(robot.p.x, robot.p.y)

            # Check collisions with enemy robots
            for enemy_id, enemy in game.enemy_robots.items():
                enemy_pos = Vector2D(enemy.p.x, enemy.p.y)
                distance = robot_pos.distance_to(enemy_pos)
                self.min_distance = min(self.min_distance, distance)

                if distance < self.scenario.collision_threshold:
                    self.collision_detected = True
                    self.collision_count += 1
                    return TestingStatus.FAILURE

            # Check collisions between friendly robots
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

            # Check if this robot reached its target
            if robot_id < len(self.scenario.friendly_targets):
                target_pos = Vector2D(*self.scenario.friendly_targets[robot_id])
                if robot_pos.distance_to(target_pos) <= self.scenario.endpoint_tolerance:
                    self.robots_reached.add(("friendly", robot_id))

        # Check if enemy robots reached their targets
        for enemy_id, enemy in game.enemy_robots.items():
            enemy_pos = Vector2D(enemy.p.x, enemy.p.y)
            if enemy_id < len(self.scenario.enemy_targets):
                target_pos = Vector2D(*self.scenario.enemy_targets[enemy_id])
                if enemy_pos.distance_to(target_pos) <= self.scenario.endpoint_tolerance:
                    self.robots_reached.add(("enemy", enemy_id))

        # Check if all robots reached their targets
        total_robots = len(self.scenario.friendly_targets) + len(self.scenario.enemy_targets)
        if len(self.robots_reached) >= total_robots:
            self.all_reached = True
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_mirror_swap(
    headless: bool,
    mode: str = "rsim",
):
    my_team_is_yellow = True
    my_team_is_right = False

    # Base positions without perturbation
    base_left = [
        (-2.5, -1.5),
        (-2.5, -0.5),
        (-2.5, 0.5),
        (-2.5, 1.5),
        (-3.5, -0.75),
        (-3.5, 0.75),
    ]

    base_right = [
        (2.5, -1.5),
        (2.5, -0.5),
        (2.5, 0.5),
        (2.5, 1.5),
        (3.5, -0.75),
        (3.5, 0.75),
    ]

    # ADDING DETERMINISTIC PERTURBATION
    # We shift the Blue team (right positions) up by exactly 2cm (0.02m).
    # This prevents perfect mathematical head-on velocity vectors.
    eps = 0.02
    left_positions = base_left
    right_positions = [(x, y + eps) for x, y in base_right]

    scenario = MultiRobotScenario(
        friendly_positions=left_positions,
        enemy_positions=right_positions,
        friendly_targets=base_right,  # Target the raw base position
        enemy_targets=base_left,  # Target the raw base position
        endpoint_tolerance=0.3,
    )

    my_strategy = MultiRobotNavigationStrategy(robot_targets={i: base_right[i] for i in range(len(left_positions))})

    opp_strategy = MultiRobotNavigationStrategy(robot_targets={i: base_left[i] for i in range(len(right_positions))})

    runner = StrategyRunner(
        strategy=my_strategy,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=6,
        exp_enemy=6,
        exp_ball=False,
        opp_strategy=opp_strategy,
    )

    test_manager = MultiRobotTestManager(scenario=scenario)
    test_passed = runner.run_test(
        test_manager=test_manager,
        episode_timeout=45.0,
        rsim_headless=headless,
    )

    assert test_passed, "Mirror charge test failed to complete"
    assert test_manager.all_reached, f"Not all robots reached targets. Reached: {len(test_manager.robots_reached)}/12"
    assert not test_manager.collision_detected, f"Robots collided! Min distance: {test_manager.min_distance:.3f}m"


def test_grid_intersection(
    headless: bool,
    mode: str = "rsim",
):
    """
    Test where two teams cross paths perpendicularly, creating 4 distinct intersection points.
    This tests dodging and path adjustment without creating an impossible 4-way center deadlock.
    """
    my_team_is_yellow = True
    my_team_is_right = False

    # Yellow team moves Left -> Right on two distinct "lanes" (closer to center)
    yellow_positions = [
        (-2.0, 0.5),
        (-2.0, -0.5),
    ]
    yellow_targets = [
        (2.0, 0.5),
        (2.0, -0.5),
    ]

    # Blue team moves Top -> Bottom on two distinct "lanes" (closer to center)
    blue_positions = [
        (0.5, 2.0),
        (-0.5, 2.0),
    ]
    blue_targets = [
        (0.5, -2.0),
        (-0.5, -2.0),
    ]

    scenario = MultiRobotScenario(
        friendly_positions=yellow_positions,
        enemy_positions=blue_positions,
        friendly_targets=yellow_targets,
        enemy_targets=blue_targets,
        endpoint_tolerance=0.25,
    )

    my_strategy = MultiRobotNavigationStrategy(
        robot_targets={i: yellow_targets[i] for i in range(len(yellow_positions))}
    )
    opp_strategy = MultiRobotNavigationStrategy(robot_targets={i: blue_targets[i] for i in range(len(blue_positions))})

    runner = StrategyRunner(
        strategy=my_strategy,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=2,
        exp_enemy=2,
        exp_ball=False,
        opp_strategy=opp_strategy,
    )

    test_manager = MultiRobotTestManager(scenario=scenario)
    test_passed = runner.run_test(
        test_manager=test_manager,
        episode_timeout=30.0,
        rsim_headless=headless,
    )

    assert test_passed, "Grid intersection test failed to complete"
    assert test_manager.all_reached, f"Not all robots reached targets. Reached: {len(test_manager.robots_reached)}/4"
    assert not test_manager.collision_detected, f"Robots collided! Min distance: {test_manager.min_distance:.3f}m"


def test_defensive_slalom(
    headless: bool,
    mode: str = "rsim",
):
    """
    Test where attacking robots must navigate through a staggered wall of stationary defenders.
    This proves the planner can find valid spatial corridors in cluttered environments.
    """
    my_team_is_yellow = True
    my_team_is_right = False

    # Yellow team starts on the left and wants to drive straight through
    yellow_positions = [
        (-2.5, 1.5),
        (-2.5, 0.0),
        (-2.5, -1.5),
    ]
    yellow_targets = [
        (2.5, 1.5),
        (2.5, 0.0),
        (2.5, -1.5),
    ]

    # Blue team forms a staggered defensive wall in the center
    # They are effectively stationary obstacles for this test
    blue_positions = [
        (0.0, 1.0),
        (0.0, -1.0),
        (-0.5, 0.0),  # Pushed slightly forward into the attacking path
        (0.5, 2.0),  # Outside blocks
        (0.5, -2.0),
    ]
    # Blue targets are their starting positions (they stay still)
    blue_targets = blue_positions.copy()

    scenario = MultiRobotScenario(
        friendly_positions=yellow_positions,
        enemy_positions=blue_positions,
        friendly_targets=yellow_targets,
        enemy_targets=blue_targets,
        endpoint_tolerance=0.25,
    )

    my_strategy = MultiRobotNavigationStrategy(
        robot_targets={i: yellow_targets[i] for i in range(len(yellow_positions))}
    )
    opp_strategy = MultiRobotNavigationStrategy(robot_targets={i: blue_targets[i] for i in range(len(blue_positions))})

    runner = StrategyRunner(
        strategy=my_strategy,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=3,
        exp_enemy=5,
        exp_ball=False,
        opp_strategy=opp_strategy,
    )

    test_manager = MultiRobotTestManager(scenario=scenario)
    test_passed = runner.run_test(
        test_manager=test_manager,
        episode_timeout=35.0,
        rsim_headless=headless,
    )

    assert test_passed, "Defensive slalom test failed to complete"
    assert test_manager.all_reached, f"Not all robots reached targets. Reached: {len(test_manager.robots_reached)}/8"
    assert not test_manager.collision_detected, f"Robots collided! Min distance: {test_manager.min_distance:.3f}m"
