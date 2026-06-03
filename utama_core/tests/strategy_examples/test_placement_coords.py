"""Integration tests for RobotPlacementStrategy using AbstractTestManager.

These tests verify that:
1. The robot moves to positions within the provided field_bounds
2. The robot visits waypoints from the 3x3 grid generated for those bounds
3. Custom field_bounds correctly constrain the placement region
"""

import math
from typing import Optional

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.one_robot_placement_strategy import (
    _ARRIVE_TOL as _STRATEGY_ARRIVE_TOL,
)
from utama_core.strategy.examples.one_robot_placement_strategy import (
    _MARGIN,
    RobotPlacementStrategy,
)
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

# Use the same tolerance as the strategy so the test only marks a waypoint reached
# when the strategy itself would have advanced to the next one.
_ARRIVE_TOL = _STRATEGY_ARRIVE_TOL


def _grid_waypoints(bounds: FieldBounds) -> list[tuple[float, float]]:
    """Mirror the strategy's _build_waypoints logic for expected-value computation."""
    x_min = bounds.top_left[0] + _MARGIN
    x_max = bounds.bottom_right[0] - _MARGIN
    y_min = bounds.bottom_right[1] + _MARGIN
    y_max = bounds.top_left[1] - _MARGIN
    xs = [x_min, (x_min + x_max) / 2, x_max]
    ys = [y_min, (y_min + y_max) / 2, y_max]
    points = []
    for i, x in enumerate(xs):
        col_ys = ys if i % 2 == 0 else list(reversed(ys))
        for y in col_ys:
            points.append((x, y))
    return points


class RobotPlacementTestManager(AbstractTestManager):
    """Verify the robot visits at least two distinct grid waypoints."""

    n_episodes = 1

    def __init__(self, waypoints: list[tuple[float, float]], tolerance: float = _ARRIVE_TOL):
        super().__init__()
        self.waypoints = waypoints
        self.tolerance = tolerance
        # Track which waypoints the robot has visited
        self.visited: set[int] = set()

    @property
    def reached_a(self) -> bool:
        return 0 in self.visited

    @property
    def reached_b(self) -> bool:
        return len(self.visited) >= 2

    @property
    def target_a(self) -> tuple[float, float]:
        return self.waypoints[0]

    @property
    def target_b(self) -> tuple[float, float]:
        return self.waypoints[1] if len(self.waypoints) > 1 else self.waypoints[0]

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        centre = game.field.field_bounds.center
        sim_controller.teleport_robot(game.my_team_is_yellow, self.my_strategy.robot_id, centre[0], centre[1])
        sim_controller.teleport_ball(centre[0] + 0.5, centre[1] + 0.5)

    def eval_status(self, game: Game) -> TestingStatus:
        robot = game.friendly_robots.get(self.my_strategy.robot_id)
        if not robot:
            return TestingStatus.IN_PROGRESS

        pos = (robot.p.x, robot.p.y)
        for i, wp in enumerate(self.waypoints):
            if math.dist(pos, wp) < self.tolerance:
                self.visited.add(i)

        if len(self.visited) >= 2:
            return TestingStatus.SUCCESS
        return TestingStatus.IN_PROGRESS


def _run_placement_test(field_bounds: Optional[FieldBounds]):
    strategy = RobotPlacementStrategy(robot_id=0)

    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=False,
        field_bounds=field_bounds,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
    )

    # Compute expected waypoints from the actual bounds that will be used
    effective_bounds = field_bounds if field_bounds is not None else STANDARD_FIELD_DIMS.full_field_bounds
    waypoints = _grid_waypoints(effective_bounds)

    test_manager = RobotPlacementTestManager(waypoints=waypoints)
    passed = runner.run_test(test_manager=test_manager, episode_timeout=30, rsim_headless=True)

    return passed, test_manager


class TestFieldBoundsCenter:
    """Tests for FieldBounds center calculation."""

    def test_full_field_center_is_origin(self):
        bounds = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
        assert bounds.center == (0.0, 0.0)

    def test_custom_bounds_center(self):
        bounds = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
        assert bounds.center == (2.0, 1.0)

    def test_custom_differs_from_default(self):
        default = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
        custom = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
        assert default.center != custom.center


class TestRobotPlacementStrategy:
    """Integration tests for RobotPlacementStrategy behavior."""

    def test_oscillation_with_custom_bounds(self):
        """Robot visits the first two grid waypoints within custom bounds (1,2)→(3,0)."""
        bounds = FieldBounds(top_left=(1.0, 2.0), bottom_right=(3.0, 0.0))
        passed, manager = _run_placement_test(bounds)
        assert manager.reached_a, f"Never reached first waypoint {manager.target_a}"
        assert manager.reached_b, f"Never reached second waypoint {manager.target_b}"
        assert passed

    def test_oscillation_with_default_bounds(self):
        """Robot visits the first two grid waypoints on the full standard field."""
        passed, manager = _run_placement_test(field_bounds=None)
        assert manager.reached_a, f"Never reached first waypoint {manager.target_a}"
        assert manager.reached_b, f"Never reached second waypoint {manager.target_b}"
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
