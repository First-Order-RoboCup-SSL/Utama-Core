"""Tests that the first game frame after reset_field reflects teleported positions."""

import os
from typing import Optional

import py_trees
import pytest

from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.run import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"

_TELEPORT_CASES = [
    # (rid, x, y, theta)
    (0, 1.0, 1.0, 0.0),
    (0, -1.0, -1.0, 1.5707),
    (0, 3.0, 3.0, 3.1415),
    (1, 2.0, -1.5, 0.0),
    (1, -2.0, 1.5, 1.5707),
]

POSITION_TOLERANCE = 0.15  # metres — accounts for one frame of physics settling


class _IdleStrategy(AbstractStrategy):
    """Minimal strategy that does nothing — used when we only care about game state."""

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.behaviours.Success(name="Idle")

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        return None


class _TeleportAccuracyTestManager(AbstractTestManager):
    n_episodes = 1

    def __init__(self, robot_id: int, target_x: float, target_y: float, target_theta: float):
        super().__init__()
        self.robot_id = robot_id
        self.target_x = target_x
        self.target_y = target_y
        self.target_theta = target_theta
        self.first_frame_position = None

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        # Park unused robots in-field (bottom-left corner, spaced apart so all are visible)
        for rid in game.friendly_robots:
            if rid != self.robot_id:
                sim_controller.teleport_robot(game.my_team_is_yellow, rid, -4.0, -2.0 + rid * 0.5)
        # Place target robot at desired position
        sim_controller.teleport_robot(
            game.my_team_is_yellow,
            self.robot_id,
            self.target_x,
            self.target_y,
            self.target_theta,
        )
        # Ball must be in-bounds so GameGater can complete
        sim_controller.teleport_ball(0.0, 0.0)

    def eval_status(self, game: Game) -> TestingStatus:
        if self.first_frame_position is None:
            robot = game.friendly_robots.get(self.robot_id)
            if robot is not None:
                self.first_frame_position = (robot.p.x, robot.p.y)
        return TestingStatus.SUCCESS


def _run_teleport_accuracy_test(robot_id: int, x: float, y: float, theta: float):
    n_robots = max(robot_id + 1, 1)
    test_manager = _TeleportAccuracyTestManager(robot_id, x, y, theta)

    runner = StrategyRunner(
        strategy=_IdleStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=n_robots,
        exp_enemy=0,
    )

    passed = runner.run_test(
        test_manager=test_manager,
        episode_timeout=5.0,
        rsim_headless=True,
    )

    print("hello!")
    assert passed, "Test episode did not complete."
    print("hello!")
    assert test_manager.first_frame_position is not None, f"Robot {robot_id} was never seen in the first game frame."

    actual_x, actual_y = test_manager.first_frame_position
    assert abs(actual_x - x) <= POSITION_TOLERANCE, (
        f"Robot {robot_id} first-frame x={actual_x:.3f} deviates from teleport target x={x:.3f} "
        f"by {abs(actual_x - x):.3f}m (tolerance={POSITION_TOLERANCE}m)"
    )
    assert abs(actual_y - y) <= POSITION_TOLERANCE, (
        f"Robot {robot_id} first-frame y={actual_y:.3f} deviates from teleport target y={y:.3f} "
        f"by {abs(actual_y - y):.3f}m (tolerance={POSITION_TOLERANCE}m)"
    )


@pytest.mark.parametrize("rid,x,y,theta", _TELEPORT_CASES)
def test_first_frame_reflects_teleported_position(rid: int, x: float, y: float, theta: float):
    """First game frame after reset_field must show the robot at its teleported position."""
    _run_teleport_accuracy_test(rid, x, y, theta)


def test_teleport_ball_first_frame():
    """First game frame after reset_field must show the ball at its teleported position."""
    target_bx, target_by = 1.5, -1.0
    first_ball_pos = {}

    class _BallTeleportManager(AbstractTestManager):
        n_episodes = 1

        def reset_field(self, sim_controller: AbstractSimController, game: Game):
            # Park robot 0 in-bounds away from the ball target
            sim_controller.teleport_robot(game.my_team_is_yellow, 0, -4.0, -2.0, 0.0)
            sim_controller.teleport_ball(target_bx, target_by)

        def eval_status(self, game: Game) -> TestingStatus:
            if "pos" not in first_ball_pos and game.ball is not None:
                first_ball_pos["pos"] = (game.ball.p.x, game.ball.p.y)
            return TestingStatus.SUCCESS

    runner = StrategyRunner(
        strategy=_IdleStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
    )

    passed = runner.run_test(
        test_manager=_BallTeleportManager(),
        episode_timeout=5.0,
        rsim_headless=True,
    )

    assert passed
    assert "pos" in first_ball_pos, "Ball was never seen in the first game frame."

    actual_bx, actual_by = first_ball_pos["pos"]
    assert (
        abs(actual_bx - target_bx) <= POSITION_TOLERANCE
    ), f"Ball first-frame x={actual_bx:.3f} deviates from teleport target x={target_bx:.3f}"
    assert (
        abs(actual_by - target_by) <= POSITION_TOLERANCE
    ), f"Ball first-frame y={actual_by:.3f} deviates from teleport target y={target_by:.3f}"
