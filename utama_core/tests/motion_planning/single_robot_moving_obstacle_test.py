from dataclasses import dataclass, field

import pytest

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
from utama_core.tests.motion_planning.strategies.oscillating_obstacle_strategy import (
    OscillatingObstacleStrategy,
)
from utama_core.tests.motion_planning.strategies.simple_navigation_strategy import (
    SimpleNavigationStrategy,
)


@dataclass
class MovingObstacleConfig:
    """Configuration for a single moving obstacle."""

    center_position: tuple[float, float]
    oscillation_axis: str  # 'x' or 'y'
    amplitude: float  # How far to move from center
    directionUpOrRight: bool  # True for going up/right first, False for going down/left first
    speed: float  # Speed of oscillation (higher = faster)


@dataclass
class MovingObstacleScenario:
    """Configuration for moving obstacle avoidance test."""

    start_position: tuple[float, float]
    target_position: tuple[float, float]
    moving_obstacles: list[MovingObstacleConfig] = field(default_factory=list)
    endpoint_tolerance: float = 0.15
    collision_threshold: float = ROBOT_RADIUS * 2.0


class MovingObstacleTestManager(AbstractTestManager):
    """Test manager that validates dynamic obstacle avoidance and target completion."""

    def __init__(self, scenario: MovingObstacleScenario, robot_id: int):
        super().__init__()
        self.scenario = scenario
        self.robot_id = robot_id
        self.n_episodes = 1
        self.endpoint_reached = False
        self.collision_detected = False
        self.min_obstacle_distance = float("inf")
        self.collision_count = 0

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        """Reset field with robot at start position and moving obstacles."""
        ini_yellow, ini_blue = map_left_right_to_colors(
            game.my_team_is_yellow,
            game.my_team_is_right,
            RIGHT_START_ONE,
            LEFT_START_ONE,
        )

        y_robots, b_robots = map_friendly_enemy_to_colors(
            game.my_team_is_yellow, game.friendly_robots, game.enemy_robots
        )

        # Teleport ALL friendly robots off-field
        for i in range(6):
            if i == self.robot_id:
                # Place the test robot at start position
                sim_controller.teleport_robot(
                    game.my_team_is_yellow,
                    i,
                    self.scenario.start_position[0],
                    self.scenario.start_position[1],
                    0.0,
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

        # Place enemy robots at their center positions (they will start moving via their strategy)
        for i in range(6):
            if i < len(self.scenario.moving_obstacles):
                obstacle_config = self.scenario.moving_obstacles[i]
                sim_controller.teleport_robot(
                    not game.my_team_is_yellow,
                    i,
                    obstacle_config.center_position[0],
                    obstacle_config.center_position[1],
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
        self.collision_count = 0

    def eval_status(self, game: Game):
        """Evaluate collision status and goal achievement."""
        robot = game.friendly_robots[self.robot_id]
        robot_position = Vector2D(robot.p.x, robot.p.y)

        # Check for collisions with moving obstacles (enemy robots)
        for obstacle_id, obstacle in game.enemy_robots.items():
            obstacle_position = Vector2D(obstacle.p.x, obstacle.p.y)
            distance = robot_position.distance_to(obstacle_position)
            self.min_obstacle_distance = min(self.min_obstacle_distance, distance)

            if distance < self.scenario.collision_threshold:
                self.collision_detected = True
                self.collision_count += 1
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
    "obstacle_scenario",
    [
        # 3 obstacles - Level 1
        {
            "start": (-3.0, 0.0),
            "target": (3.0, 0.0),
            "obstacles": [
                MovingObstacleConfig((-1.5, 0.0), "y", 1.5, True, 2.0),
                MovingObstacleConfig((0.0, 0.0), "x", 0.8, True, 1.2),
                MovingObstacleConfig((1.5, 0.0), "y", 1.0, False, 1.5),
            ],
        },
        # Fast moving vertical obstacles - Level 2
        {
            "start": (-3.0, 0.0),
            "target": (3.0, 0.0),
            "obstacles": [
                MovingObstacleConfig((-1.0, 0.5), "y", 1.5, True, 2.5),
                MovingObstacleConfig((0.0, -0.5), "y", 1.5, False, 2.0),
                MovingObstacleConfig((1.0, 0.0), "y", 1.2, True, 2.0),
                MovingObstacleConfig((2.0, 0.0), "y", 1.2, False, 1.5),
            ],
        },
        # Mixed speed obstacles - Level 3 (max 6 robots)
        {
            "start": (-3.0, 0.0),
            "target": (3.0, 0.0),
            "obstacles": [
                MovingObstacleConfig((-2.0, 0.5), "y", 1.2, True, 2.0),  # Two vertical + 1 horizontal lines
                MovingObstacleConfig((-0.5, -0.5), "y", 1.2, False, 2.0),
                MovingObstacleConfig((0.0, 1.0), "x", 0.8, True, 1.5),
                MovingObstacleConfig((1.0, 0.0), "y", 1.0, True, 2.5),  # One vertical + 2 horizontal lines near target
                MovingObstacleConfig((2.2, 0.5), "x", 0.6, False, 1.3),
                MovingObstacleConfig((2.2, -0.5), "x", 0.6, True, 1.3),
            ],
        },
    ],
)
def test_single_robot_moving_obstacles(
    headless: bool,
    obstacle_scenario: dict,
    mode: str = "rsim",
):
    """
    Test that a robot navigates from start to target while avoiding moving obstacles.

    The robot should:
    1. Navigate from start position to target position
    2. Avoid multiple enemy robots that oscillate back and forth
    3. Maintain safe distance from all moving obstacles
    4. Successfully reach the target despite dynamic environment
    """
    # Hardcoded values to run test only once
    my_team_is_yellow = True
    my_team_is_right = True
    robot_id = 0

    scenario = MovingObstacleScenario(
        start_position=obstacle_scenario["start"],
        target_position=obstacle_scenario["target"],
        moving_obstacles=obstacle_scenario["obstacles"],
    )

    # Create friendly robot strategy
    my_strategy = SimpleNavigationStrategy(
        robot_id=robot_id,
        target_position=scenario.target_position,
        target_orientation=0.0,
    )

    # Create opponent strategy with oscillating obstacles
    opp_strategy = OscillatingObstacleStrategy(
        obstacle_configs=scenario.moving_obstacles,
    )

    runner = StrategyRunner(
        strategy=my_strategy,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=1,
        exp_enemy=len(scenario.moving_obstacles),
        opp_strategy=opp_strategy,
        control_scheme="dwa",
        opp_control_scheme="pid",  # Use PID so obstacles follow exact paths without avoiding the robot
    )

    test_manager = MovingObstacleTestManager(scenario=scenario, robot_id=robot_id)
    test_passed = runner.run_test(
        testManager=test_manager,
        episode_timeout=30.0,  # Longer timeout for moving obstacles
        rsim_headless=headless,
    )

    # Assertions
    assert test_passed, "Moving obstacle avoidance test failed to complete"
    assert test_manager.endpoint_reached, f"Robot failed to reach target {scenario.target_position}"
    assert not test_manager.collision_detected, (
        f"Robot collided with moving obstacle {test_manager.collision_count} time(s)! "
        f"Minimum distance: {test_manager.min_obstacle_distance:.3f}m "
        f"(threshold: {scenario.collision_threshold:.3f}m)"
    )
    assert test_manager.min_obstacle_distance >= scenario.collision_threshold, (
        f"Robot got too close to moving obstacles: {test_manager.min_obstacle_distance:.3f}m "
        f"(minimum safe distance: {scenario.collision_threshold:.3f}m)"
    )
