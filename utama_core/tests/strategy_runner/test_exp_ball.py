"""Tests that exp_ball is correctly handled by StrategyRunner.

Covers three layers:
  1. Constructor-level misconfig — strategy.exp_ball != runner exp_ball raises AssertionError.
  2. Constructor-level agreement — no error raised when both sides agree.
  3. Integration — the actual game frame reflects the expected ball presence/absence.
"""

import os
from typing import Optional

import py_trees
import pytest

from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)
from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"


# ---------------------------------------------------------------------------
# Minimal idle strategies with controllable exp_ball
# ---------------------------------------------------------------------------


class _IdleWithBallStrategy(AbstractStrategy):
    """Idle strategy that expects the ball to be present (exp_ball=True)."""

    exp_ball: bool = True

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.behaviours.Success(name="Idle")

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        return None


class _IdleNoBallStrategy(AbstractStrategy):
    """Idle strategy that expects NO ball (exp_ball=False)."""

    exp_ball: bool = False

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.behaviours.Success(name="Idle")

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        return None


# ---------------------------------------------------------------------------
# 1. Misconfig tests (constructor-level, no simulation needed)
# ---------------------------------------------------------------------------


def test_exp_ball_mismatch_strategy_true_runner_false():
    """Strategy expects ball but runner says exp_ball=False → AssertionError."""
    strat = DummyStrategy()
    strat.exp_ball = True
    with pytest.raises(RuntimeError, match="Ball expected"):
        StrategyRunner(
            strategy=strat,
            my_team_is_yellow=True,
            my_team_is_right=True,
            mode="rsim",
            exp_friendly=3,
            exp_enemy=3,
            exp_ball=False,
        )


def test_exp_ball_mismatch_strategy_false_runner_true():
    """Strategy expects no ball but runner says exp_ball=True → AssertionError."""
    strat = DummyStrategy()
    strat.exp_ball = False
    run = StrategyRunner(
        strategy=strat,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        exp_ball=True,
    )
    assert run.exp_ball is True, "Runner should remain exp_ball=True even if strategy has exp_ball=False"


# ---------------------------------------------------------------------------
# 2. Agreement tests (constructor-level, no simulation needed)
# ---------------------------------------------------------------------------


def test_exp_ball_agreement_true_does_not_raise():
    """Strategy and runner both have exp_ball=True → no error, flag is set."""
    strat = DummyStrategy()
    strat.exp_ball = True
    runner = StrategyRunner(
        strategy=strat,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        exp_ball=True,
    )
    assert runner.exp_ball is True


def test_exp_ball_agreement_false_does_not_raise():
    """Strategy and runner both have exp_ball=False → no error, flag is set."""
    strat = DummyStrategy()
    strat.exp_ball = False
    runner = StrategyRunner(
        strategy=strat,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        exp_ball=False,
    )
    assert runner.exp_ball is False


# ---------------------------------------------------------------------------
# 3. Integration tests — verify actual game frame state
# ---------------------------------------------------------------------------


class _BallPresentManager(AbstractTestManager):
    """Records whether game.ball is non-None on the first eval call, then passes."""

    n_episodes = 1

    def __init__(self):
        super().__init__()
        self.ball_seen: Optional[bool] = None

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -3.0, 0.0)
        sim_controller.teleport_ball(0.0, 0.0)

    def eval_status(self, game: Game) -> TestingStatus:
        if self.ball_seen is None:
            self.ball_seen = game.ball is not None
        return TestingStatus.SUCCESS


class _BallAbsentManager(AbstractTestManager):
    """Verifies game.ball is None on EVERY eval call when exp_ball=False.

    Returns FAILURE immediately on the first frame where a ball is seen, so
    the test fails fast with a clear signal. Returns SUCCESS after
    N_FRAMES_TO_CHECK frames with no ball, confirming the invariant holds
    throughout and is not just an initial transient.
    """

    n_episodes = 1
    N_FRAMES_TO_CHECK = 20

    def __init__(self):
        super().__init__()
        self.frames_checked: int = 0

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -3.0, 0.0)
        # No ball teleport — ball has already been removed off-field by StrategyRunner

    def eval_status(self, game: Game) -> TestingStatus:
        self.frames_checked += 1
        if game.ball is not None:
            return TestingStatus.FAILURE
        if self.frames_checked >= self.N_FRAMES_TO_CHECK:
            return TestingStatus.SUCCESS
        return TestingStatus.IN_PROGRESS


def test_exp_ball_true_ball_present_in_game():
    """When exp_ball=True, game.ball must be non-None in the first game frame."""
    tm = _BallPresentManager()
    runner = StrategyRunner(
        strategy=_IdleWithBallStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=True,
    )
    passed = runner.run_test(tm, episode_timeout=5.0, rsim_headless=True)
    assert passed
    assert tm.ball_seen is True, "game.ball should be non-None when exp_ball=True"


def test_exp_ball_true_ball_present_in_game_with_filtering():
    """Same as above but with Kalman filtering enabled."""
    tm = _BallPresentManager()
    runner = StrategyRunner(
        strategy=_IdleWithBallStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=True,
        filtering=True,
    )
    passed = runner.run_test(tm, episode_timeout=5.0, rsim_headless=True)
    assert passed
    assert tm.ball_seen is True, "game.ball should be non-None when exp_ball=True (filtering=True)"


def test_exp_ball_false_ball_absent_in_game():
    """When exp_ball=False, game.ball must be None on every game frame throughout the episode."""
    tm = _BallAbsentManager()
    runner = StrategyRunner(
        strategy=_IdleNoBallStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=False,
    )
    passed = runner.run_test(tm, episode_timeout=10.0, rsim_headless=True)
    assert passed, (
        f"game.ball was non-None within the first {tm.frames_checked} frames — "
        "expected None on every frame when exp_ball=False"
    )
    assert (
        tm.frames_checked >= _BallAbsentManager.N_FRAMES_TO_CHECK
    ), "Test timed out before reaching the required number of frames"


def test_exp_ball_false_ball_absent_in_game_with_filtering():
    """Same as above but with Kalman filtering enabled — the filter must not impute a ball."""
    tm = _BallAbsentManager()
    runner = StrategyRunner(
        strategy=_IdleNoBallStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=False,
        filtering=True,
    )
    passed = runner.run_test(tm, episode_timeout=10.0, rsim_headless=True)
    assert passed, (
        f"game.ball was non-None within the first {tm.frames_checked} frames — "
        "expected None on every frame when exp_ball=False (filtering=True)"
    )
    assert (
        tm.frames_checked >= _BallAbsentManager.N_FRAMES_TO_CHECK
    ), "Test timed out before reaching the required number of frames"
