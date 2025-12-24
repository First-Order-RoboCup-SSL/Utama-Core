from dataclasses import dataclass, field

import pytest

from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.run import StrategyRunner
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)
from utama_core.tests.motion_planning.strategies.simple_navigation_strategy import (
    SimpleNavigationStrategy,
)


@dataclass
class CollisionAvoidanceScenario:
    """Configuration for a collision avoidance test."""

    start_position: tuple[float, float]
    target_position: tuple[float, float]
    obstacle_positions: list[tuple[float, float]] = field(default_factory=list)
    endpoint_tolerance: float = 0.15
    collision_threshold: float = ROBOT_RADIUS * 2.0  # Two robot radii for safety margin


class CollisionAvoidanceTestManager(AbstractTestManager):
    """Test manager that validates obstacle avoidance and target completion."""

    def __init__(self, scenario: CollisionAvoidanceScenario, robot_id: int):
        super().__init__()
        self.scenario = scenario
        self.robot_id = robot_id
        self.n_episodes = 1
        self.endpoint_reached = False
        self.collision_detected = False
        self.min_obstacle_distance = float("inf")

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """Reset field with robot at start position and obstacles along the path."""
        # Teleport ALL friendly robots off-field first (to clean up from previous tests)
        for i in range(6):  # SSL has max 6 robots per team
            if i == self.robot_id:
                # Place the test robot at start position
                sim_controller.teleport_robot(
                    game.my_team_is_yellow,
                    i,
                    self.scenario.start_position[0],
                    self.scenario.start_position[1],
                    0.0,  # Face forward
                )
            else:
                # Move other friendly robots far away
                sim_controller.teleport_robot(
                    game.my_team_is_yellow,
                    i,
                    -10.0,
                    -10.0,
                    0.0,
                )

        # Place enemy robots as obstacles
        for i in range(6):  # Handle all possible enemy robots
            if i < len(self.scenario.obstacle_positions):
                x, y = self.scenario.obstacle_positions[i]
                sim_controller.teleport_robot(
                    not game.my_team_is_yellow,
                    i,
                    x,
                    y,
                    0.0,
                )
            else:
                # Move extra enemy robots far away
                sim_controller.teleport_robot(
                    not game.my_team_is_yellow,
                    i,
                    -10.0,
                    -10.0,
                    0.0,
                )

        # Place ball at target (for visual reference)
        sim_controller.teleport_ball(
            self.scenario.target_position[0],
            self.scenario.target_position[1],
        )

        self._reset_metrics()

    def _reset_metrics(self):
        """Reset tracking metrics for new episode."""
        self.endpoint_reached = False
        self.collision_detected = False
        self.min_obstacle_distance = float("inf")

    def eval_status(self, game: Game):
        """Evaluate collision status and goal achievement."""
        robot = game.friendly_robots[self.robot_id]
        robot_position = Vector2D(robot.p.x, robot.p.y)

        # Check for collisions with obstacles (enemy robots)
        for obstacle_id, obstacle in game.enemy_robots.items():
            obstacle_position = Vector2D(obstacle.p.x, obstacle.p.y)
            distance = robot_position.distance_to(obstacle_position)
            self.min_obstacle_distance = min(self.min_obstacle_distance, distance)

            if distance < self.scenario.collision_threshold:
                self.collision_detected = True
                return TestingStatus.FAILURE

        # Check if target reached
        target = Vector2D(self.scenario.target_position[0], self.scenario.target_position[1])
        if robot_position.distance_to(target) <= self.scenario.endpoint_tolerance:
            self.endpoint_reached = True
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS

    def get_n_episodes(self):
        return self.n_episodes


@pytest.mark.parametrize(
    "obstacle_config",
    [
        # Single obstacle in the middle
        {
            "start": (-3.0, 0.0),
            "target": (3.0, 0.0),
            "obstacles": [(0.0, 0.0)],
        },
        # Three obstacles creating a narrow gap
        {
            "start": (-3.0, 0.0),
            "target": (3.0, 0.0),
            "obstacles": [(-1.0, 0.3), (0.0, 0.0), (1.0, -0.3)],
        },
        # Six obstacles requiring path planning
        {
            "start": (-3.0, 0.0),
            "target": (3.0, 0.0),
            "obstacles": [
                (-1.5, 0.0),
                (0.0, -0.5),
                (0.0, 0.5),
                (1.5, -1.0),
                (1.5, 0.0),
                (1.5, 1.0),
            ],
        },
    ],
)
def test_collision_avoidance_goal_to_goal(
    headless: bool,
    obstacle_config: dict,
    mode: str = "rsim",
):
    """
    Test that robot navigates from one goal to another while avoiding obstacles.

    The robot should:
    1. Start near one goal
    2. Navigate to the opposite goal
    3. Avoid all obstacles along the straight-line path
    4. Maintain safe distance from obstacles
    5. Successfully reach the target position
    """
    # Hardcoded values to run test only once
    my_team_is_yellow = True
    my_team_is_right = True
    robot_id = 0

    scenario = CollisionAvoidanceScenario(
        start_position=obstacle_config["start"],
        target_position=obstacle_config["target"],
        obstacle_positions=obstacle_config["obstacles"],
    )

    # Use RobotPlacementStrategy which will move robot to target
    # We need to modify it slightly for this test
    runner = StrategyRunner(
        strategy=SimpleNavigationStrategy(
            robot_id=robot_id,
            target_position=scenario.target_position,
            target_orientation=0.0,
        ),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=1,
        exp_enemy=len(obstacle_config["obstacles"]),
        control_scheme="dwa",  # Use PID for obstacle avoidance
    )

    test_manager = CollisionAvoidanceTestManager(scenario=scenario, robot_id=robot_id)
    test_passed = runner.run_test(
        testManager=test_manager,
        episode_timeout=20.0,  # Give enough time to navigate
        rsim_headless=headless,
    )

    # Assertions
    assert test_passed, "Collision avoidance test failed to complete"
    assert test_manager.endpoint_reached, f"Robot failed to reach target {scenario.target_position}"
    assert not test_manager.collision_detected, (
        f"Robot collided with obstacle! Minimum distance: {test_manager.min_obstacle_distance:.3f}m "
        f"(threshold: {scenario.collision_threshold:.3f}m)"
    )
    assert test_manager.min_obstacle_distance >= scenario.collision_threshold, (
        f"Robot got too close to obstacles: {test_manager.min_obstacle_distance:.3f}m "
        f"(minimum safe distance: {scenario.collision_threshold:.3f}m)"
    )


def test_simple_straight_line_no_obstacles(
    headless: bool,
    mode: str = "rsim",
):
    """Baseline test: robot should reach goal with no obstacles."""
    # Hardcoded values to run test only once
    my_team_is_yellow = True
    my_team_is_right = True
    robot_id = 0

    scenario = CollisionAvoidanceScenario(
        start_position=(-3.0, 0.0),
        target_position=(3.0, 0.0),
        obstacle_positions=[],  # No obstacles
    )

    runner = StrategyRunner(
        strategy=SimpleNavigationStrategy(
            robot_id=robot_id,
            target_position=scenario.target_position,
            target_orientation=0.0,
        ),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=1,
        exp_enemy=0,
        control_scheme="dwa",
    )

    test_manager = CollisionAvoidanceTestManager(scenario=scenario, robot_id=robot_id)
    test_passed = runner.run_test(
        testManager=test_manager,
        episode_timeout=15.0,
        rsim_headless=headless,
    )

    assert test_passed, "Simple straight-line navigation failed"
    assert test_manager.endpoint_reached, "Robot failed to reach target with no obstacles"
