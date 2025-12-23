"""Tests for multiple robots collision avoidance scenarios."""

import os
from dataclasses import dataclass


from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.global_utils.mapping_utils import (
    map_friendly_enemy_to_colors,
    map_left_right_to_colors,
)
from utama_core.run import StrategyRunner
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)
from utama_core.tests.motion_planning.strategies.simple_navigation_strategy import (
    SimpleNavigationStrategy,
)

# Fix pygame window position for screen capture
os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"


@dataclass
class MirrorChargeConfig:
    """Configuration for mirror positions where teams charge at each other."""

    positions: list[tuple[float, float]]  # Positions for one team (left side)
    collision_threshold: float = ROBOT_RADIUS * 2.0
    endpoint_tolerance: float = 0.2


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

    def __init__(self, scenario: MultiRobotScenario):
        super().__init__()
        self.scenario = scenario
        self.n_episodes = 1
        self.all_reached = False
        self.collision_detected = False
        self.min_distance = float("inf")
        self.collision_count = 0
        self.robots_reached = set()

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """Reset field with all robots at their starting positions."""
        # Teleport friendly robots to starting positions
        for i, (x, y) in enumerate(self.scenario.friendly_positions):
            if i < 6:  # Max 6 robots per team
                sim_controller.teleport_robot(
                    game.my_team_is_yellow,
                    i,
                    x,
                    y,
                    0.0,
                )

        # Teleport remaining friendly robots far away
        for i in range(len(self.scenario.friendly_positions), 6):
            sim_controller.teleport_robot(
                game.my_team_is_yellow,
                i,
                -10.0,
                -10.0,
                0.0,
            )

        # Teleport enemy robots to starting positions
        for i, (x, y) in enumerate(self.scenario.enemy_positions):
            if i < 6:
                sim_controller.teleport_robot(
                    not game.my_team_is_yellow,
                    i,
                    x,
                    y,
                    0.0,
                )

        # Teleport remaining enemy robots far away
        for i in range(len(self.scenario.enemy_positions), 6):
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
        self.all_reached = False
        self.collision_detected = False
        self.min_distance = float("inf")
        self.collision_count = 0
        self.robots_reached = set()

    def eval_status(self, game: Game):
        """Evaluate collision status and goal achievement for all robots."""
        # Check collisions between all pairs of robots (friendly-enemy and friendly-friendly)
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

    def get_n_episodes(self):
        return self.n_episodes


def test_mirror_charge_head_on(
    headless: bool,
    mode: str = "rsim",
):
    """
    Test where two teams of 6 robots charge straight at each other from mirror positions.

    The robots should:
    1. Start at mirror positions on opposite sides of the field
    2. Navigate to their mirror counterparts' starting positions
    3. Avoid collisions with all robots (teammates and opponents)
    4. Successfully reach their target positions
    """
    my_team_is_yellow = True
    my_team_is_right = False  # Yellow on left, Blue on right

    # Define mirror positions (6 robots in formation)
    # Left side positions (Yellow team starting positions)
    left_positions = [
        (-2.5, -1.5),  # Bottom row
        (-2.5, -0.5),
        (-2.5, 0.5),
        (-2.5, 1.5),  # Top row
        (-3.5, -0.75),  # Back row
        (-3.5, 0.75),
    ]

    # Right side positions (Blue team starting positions - mirrors of left)
    right_positions = [
        (2.5, -1.5),
        (2.5, -0.5),
        (2.5, 0.5),
        (2.5, 1.5),
        (3.5, -0.75),
        (3.5, 0.75),
    ]

    scenario = MultiRobotScenario(
        friendly_positions=left_positions,  # Yellow starts on left
        enemy_positions=right_positions,  # Blue starts on right
        friendly_targets=right_positions,  # Yellow targets are Blue's starting positions
        enemy_targets=left_positions,  # Blue targets are Yellow's starting positions
        endpoint_tolerance=0.3,
    )

    # Create strategies for both teams
    # Each robot navigates to its mirror position on the opposite side
    friendly_strategies = []
    for robot_id in range(len(left_positions)):
        friendly_strategies.append(
            SimpleNavigationStrategy(
                robot_id=robot_id,
                target_position=right_positions[robot_id],
                target_orientation=0.0,
            )
        )

    enemy_strategies = []
    for robot_id in range(len(right_positions)):
        enemy_strategies.append(
            SimpleNavigationStrategy(
                robot_id=robot_id,
                target_position=left_positions[robot_id],
                target_orientation=0.0,
            )
        )

    # For simplicity, use a parallel strategy that runs all robot strategies
    # Since we need a single strategy object, we'll create a custom one
    from utama_core.tests.motion_planning.strategies.multi_robot_navigation_strategy import (
        MultiRobotNavigationStrategy,
    )

    my_strategy = MultiRobotNavigationStrategy(
        robot_targets={i: right_positions[i] for i in range(len(left_positions))}
    )

    opp_strategy = MultiRobotNavigationStrategy(
        robot_targets={i: left_positions[i] for i in range(len(right_positions))}
    )

    runner = StrategyRunner(
        strategy=my_strategy,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=6,
        exp_enemy=6,
        opp_strategy=opp_strategy,
        control_scheme="dwa",  # Both teams use DWA for collision avoidance
    )

    test_manager = MultiRobotTestManager(scenario=scenario)
    test_passed = runner.run_test(
        testManager=test_manager,
        episode_timeout=45.0,  # Longer timeout for 12 robots
        rsim_headless=headless,
    )

    # Assertions
    assert test_passed, "Mirror charge test failed to complete"
    assert test_manager.all_reached, (
        f"Not all robots reached their targets. "
        f"Reached: {len(test_manager.robots_reached)}/{len(scenario.friendly_targets) + len(scenario.enemy_targets)}"
    )
    assert not test_manager.collision_detected, (
        f"Robots collided {test_manager.collision_count} time(s)! "
        f"Minimum distance: {test_manager.min_distance:.3f}m "
        f"(threshold: {scenario.collision_threshold:.3f}m)"
    )
    assert test_manager.min_distance >= scenario.collision_threshold, (
        f"Robots got too close: {test_manager.min_distance:.3f}m "
        f"(minimum safe distance: {scenario.collision_threshold:.3f}m)"
    )


def test_diagonal_cross_square(
    headless: bool,
    mode: str = "rsim",
):
    """
    Test where 4 robots (2 per team) start at corners of a square and cross diagonally.

    The robots should:
    1. Start at the 4 corners of a square
    2. Navigate diagonally to the opposite corner
    3. Avoid collisions at the center where all paths cross
    4. Successfully reach their target positions
    """
    my_team_is_yellow = True
    my_team_is_right = False

    # Define square corners (2m x 2m square centered at origin)
    # Top-left and bottom-right for Yellow team
    yellow_positions = [
        (-1.5, 1.5),  # Top-left corner (robot 0)
        (1.5, -1.5),  # Bottom-right corner (robot 1)
    ]

    # Top-right and bottom-left for Blue team
    blue_positions = [
        (1.5, 1.5),  # Top-right corner (robot 0)
        (-1.5, -1.5),  # Bottom-left corner (robot 1)
    ]

    # Each robot goes to the opposite diagonal corner
    yellow_targets = [
        (1.5, -1.5),  # Robot 0: top-left → bottom-right
        (-1.5, 1.5),  # Robot 1: bottom-right → top-left
    ]

    blue_targets = [
        (-1.5, -1.5),  # Robot 0: top-right → bottom-left
        (1.5, 1.5),  # Robot 1: bottom-left → top-right
    ]

    scenario = MultiRobotScenario(
        friendly_positions=yellow_positions,
        enemy_positions=blue_positions,
        friendly_targets=yellow_targets,
        enemy_targets=blue_targets,
        endpoint_tolerance=0.25,
    )

    from utama_core.tests.motion_planning.strategies.multi_robot_navigation_strategy import (
        MultiRobotNavigationStrategy,
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
        opp_strategy=opp_strategy,
        control_scheme="dwa",  # Use DWA for collision avoidance
    )

    test_manager = MultiRobotTestManager(scenario=scenario)
    test_passed = runner.run_test(
        testManager=test_manager,
        episode_timeout=30.0,
        rsim_headless=headless,
    )

    # Assertions
    assert test_passed, "Diagonal cross test failed to complete"
    assert test_manager.all_reached, (
        f"Not all robots reached their targets. " f"Reached: {len(test_manager.robots_reached)}/4"
    )
    assert not test_manager.collision_detected, (
        f"Robots collided {test_manager.collision_count} time(s) at center crossing! "
        f"Minimum distance: {test_manager.min_distance:.3f}m "
        f"(threshold: {scenario.collision_threshold:.3f}m)"
    )
    assert test_manager.min_distance >= scenario.collision_threshold, (
        f"Robots got too close at crossing: {test_manager.min_distance:.3f}m "
        f"(minimum safe distance: {scenario.collision_threshold:.3f}m)"
    )
