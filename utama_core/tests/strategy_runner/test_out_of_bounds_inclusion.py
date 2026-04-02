from unittest.mock import patch

import py_trees
import pytest

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)


class _IdleStrategy(AbstractStrategy):
    exp_ball: bool = True

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.behaviours.Success(name="Idle")

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self):
        return None


class _ImmediateSuccessManager(AbstractTestManager):
    n_episodes = 1

    def reset_field(self, sim_controller, game):
        return None

    def eval_status(self, game):
        return TestingStatus.SUCCESS


def _frame(out_of_bounds: bool) -> GameFrame:
    x_pos = 20.0 if out_of_bounds else 0.0
    robot = Robot(
        id=0,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(x_pos, 0.0),
        v=Vector2D(0.0, 0.0),
        a=Vector2D(0.0, 0.0),
        orientation=0.0,
    )
    ball = Ball(
        p=Vector3D(0.0, 0.0, 0.0),
        v=Vector3D(0.0, 0.0, 0.0),
        a=Vector3D(0.0, 0.0, 0.0),
    )
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={0: robot},
        enemy_robots={},
        ball=ball,
    )


def _make_runner() -> StrategyRunner:
    return StrategyRunner(
        strategy=_IdleStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=True,
    )


def test_sim_initialization_does_not_call_game_gater():
    with patch(
        "utama_core.run.strategy_runner.GameGater.wait_until_game_valid",
        side_effect=AssertionError("wait_until_game_valid should not be called during __init__"),
    ):
        runner = _make_runner()
    runner.close()


def test_run_test_allows_out_of_bounds_on_initial_setup_only():
    runner = _make_runner()
    manager = _ImmediateSuccessManager()
    include_flags: list[bool] = []

    def fake_wait(*args, **kwargs):
        include_flag = kwargs["incl_out_of_bounds_vision"]
        include_flags.append(include_flag)
        if include_flag:
            return _frame(out_of_bounds=True), None
        return _frame(out_of_bounds=False), None

    with (
        patch(
            "utama_core.run.strategy_runner.GameGater.wait_until_game_valid",
            side_effect=fake_wait,
        ),
        patch(
            "utama_core.run.strategy_runner.time.sleep",
            return_value=None,
        ),
        patch.object(
            runner,
            "_run_step",
            side_effect=lambda: None,
        ),
        patch.object(
            runner,
            "_reset_game",
            side_effect=lambda: runner._load_game(),
        ),
    ):
        passed = runner.run_test(manager, episode_timeout=1.0, rsim_headless=True)

    assert passed
    assert include_flags == [True, False]


def test_run_test_enforces_bounds_on_first_episode_load():
    runner = _make_runner()
    manager = _ImmediateSuccessManager()
    include_flags: list[bool] = []

    def fake_wait(*args, **kwargs):
        include_flag = kwargs["incl_out_of_bounds_vision"]
        include_flags.append(include_flag)
        if include_flag:
            return _frame(out_of_bounds=True), None
        raise TimeoutError("out-of-bounds vision rejected after reset")

    with (
        patch(
            "utama_core.run.strategy_runner.GameGater.wait_until_game_valid",
            side_effect=fake_wait,
        ),
        patch(
            "utama_core.run.strategy_runner.time.sleep",
            return_value=None,
        ),
        patch.object(
            runner,
            "_reset_game",
            side_effect=lambda: runner._load_game(),
        ),
    ):
        with pytest.raises(TimeoutError, match="out-of-bounds vision rejected"):
            runner.run_test(manager, episode_timeout=1.0, rsim_headless=True)

    assert include_flags == [True, False]


def test_run_enforces_bounds_on_first_load_for_sims():
    runner = _make_runner()
    include_flags: list[bool] = []

    def fake_wait(*args, **kwargs):
        include_flag = kwargs["incl_out_of_bounds_vision"]
        include_flags.append(include_flag)
        raise TimeoutError("run() should reject out-of-bounds vision on first load")

    with patch(
        "utama_core.run.strategy_runner.GameGater.wait_until_game_valid",
        side_effect=fake_wait,
    ):
        with pytest.raises(TimeoutError, match="reject out-of-bounds vision"):
            runner.run()

    assert include_flags == [False]
