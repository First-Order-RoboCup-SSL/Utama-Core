import pytest

from utama_core.config.enums import Mode
from utama_core.entities.game.field import FieldBounds
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy


@pytest.fixture
def base_runner():
    return StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
    )


def test_load_mode_valid(base_runner):
    assert base_runner._load_mode("rsim") == Mode.RSIM
    assert base_runner._load_mode("grsim") == Mode.GRSIM
    assert base_runner._load_mode("real") == Mode.REAL


def test_load_mode_invalid(base_runner):
    with pytest.raises(ValueError):
        base_runner._load_mode("invalid_mode")


def test_assert_exp_robots_valid(base_runner):
    base_runner._assert_exp_robots(DummyStrategy(), None, 3, 3)  # Should not raise


def test_assert_exp_robots_too_many_friendly(base_runner):
    with pytest.raises(AssertionError):
        base_runner._assert_exp_robots(DummyStrategy(), None, 999, 3)


def test_assert_exp_robots_too_few_friendly(base_runner):
    base_runner.exp_friendly = 0
    with pytest.raises(AssertionError):
        base_runner._assert_exp_robots(DummyStrategy(), None, 0, 3)


def test_assert_exp_goals_fails(base_runner):
    # Mock the strategy to return False on assert_exp_goals
    base_runner.my.strategy.assert_exp_goals = lambda *a, **k: False
    with pytest.raises(AssertionError):
        base_runner._assert_exp_goals()


def test_load_robot_controllers_invalid_mode(base_runner):
    base_runner.mode = None  # invalid mode
    with pytest.raises(ValueError):
        base_runner._load_robot_controllers()


def test_strategy_runner_valid_bounds():
    """Should construct cleanly with valid field bounds."""
    valid_bounds = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    runner = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        field_bounds=valid_bounds,
    )
    assert isinstance(runner, StrategyRunner)


def test_strategy_runner_invalid_bounds():
    """Should raise AssertionError when bounds are invalid."""
    invalid_bounds = FieldBounds(top_left=(3, 2), bottom_right=(-3, -2))

    with pytest.raises(AssertionError):
        StrategyRunner(
            strategy=DummyStrategy(),
            my_team_is_yellow=True,
            my_team_is_right=True,
            mode="rsim",
            exp_friendly=3,
            exp_enemy=3,
            field_bounds=invalid_bounds,
        )
