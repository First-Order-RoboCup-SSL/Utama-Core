from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utama_core.config.enums import Mode
from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
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


def test_resolve_referee_system_defaults_to_none_in_rsim():
    assert StrategyRunner._resolve_referee_system(Mode.RSIM, None, None) == "none"


def test_resolve_referee_system_defaults_to_none_in_grsim():
    assert StrategyRunner._resolve_referee_system(Mode.GRSIM, None, None) == "none"


def test_resolve_referee_system_defaults_to_none_even_when_custom_referee_is_provided():
    with pytest.raises(ValueError, match="custom_referee"):
        StrategyRunner._resolve_referee_system(Mode.RSIM, None, object())


def test_resolve_referee_system_rejects_invalid_name():
    with pytest.raises(ValueError, match="Unknown referee_system"):
        StrategyRunner._resolve_referee_system(Mode.RSIM, "auto", None)


def test_resolve_referee_system_rejects_official_in_rsim():
    with pytest.raises(ValueError, match="official"):
        StrategyRunner._resolve_referee_system(Mode.RSIM, "official", None)


def test_resolve_referee_system_requires_custom_referee_for_custom_mode():
    with pytest.raises(ValueError, match="custom_referee"):
        StrategyRunner._resolve_referee_system(Mode.RSIM, "custom", None)


def test_resolve_referee_system_rejects_custom_referee_with_none_mode():
    with pytest.raises(ValueError, match="custom_referee"):
        StrategyRunner._resolve_referee_system(Mode.RSIM, "none", object())


def test_resolve_referee_system_rejects_custom_referee_with_official_mode():
    with pytest.raises(ValueError, match="custom_referee"):
        StrategyRunner._resolve_referee_system(Mode.GRSIM, "official", object())


def test_setup_vision_and_referee_starts_vision_only_when_referee_none(monkeypatch):
    from utama_core.run import strategy_runner as runner_mod

    started = []

    class DummyVisionReceiver:
        def __init__(self, buffers):
            self.buffers = buffers

    class DummyRefereeReceiver:
        def __init__(self, buffer):
            self.buffer = buffer

    monkeypatch.setattr(runner_mod, "VisionReceiver", DummyVisionReceiver)
    monkeypatch.setattr(runner_mod, "RefereeMessageReceiver", DummyRefereeReceiver)

    fake_runner = SimpleNamespace(
        mode=Mode.GRSIM,
        referee_system="none",
        start_threads=lambda vision_receiver, referee_receiver=None: started.append(
            (vision_receiver, referee_receiver)
        ),
    )

    vision_buffers, ref_buffer = StrategyRunner._setup_vision_and_referee(fake_runner)

    assert len(vision_buffers) > 0
    assert ref_buffer.maxlen == 1
    assert len(started) == 1
    assert isinstance(started[0][0], DummyVisionReceiver)
    assert started[0][1] is None


def test_setup_vision_and_referee_starts_vision_only_when_referee_custom(monkeypatch):
    from utama_core.run import strategy_runner as runner_mod

    started = []

    class DummyVisionReceiver:
        def __init__(self, buffers):
            self.buffers = buffers

    class DummyRefereeReceiver:
        def __init__(self, buffer):
            self.buffer = buffer

    monkeypatch.setattr(runner_mod, "VisionReceiver", DummyVisionReceiver)
    monkeypatch.setattr(runner_mod, "RefereeMessageReceiver", DummyRefereeReceiver)

    fake_runner = SimpleNamespace(
        mode=Mode.REAL,
        referee_system="custom",
        start_threads=lambda vision_receiver, referee_receiver=None: started.append(
            (vision_receiver, referee_receiver)
        ),
    )

    StrategyRunner._setup_vision_and_referee(fake_runner)

    assert len(started) == 1
    assert isinstance(started[0][0], DummyVisionReceiver)
    assert started[0][1] is None


def test_setup_vision_and_referee_starts_both_receivers_when_referee_official(monkeypatch):
    from utama_core.run import strategy_runner as runner_mod

    started = []

    class DummyVisionReceiver:
        def __init__(self, buffers):
            self.buffers = buffers

    class DummyRefereeReceiver:
        def __init__(self, buffer):
            self.buffer = buffer

    monkeypatch.setattr(runner_mod, "VisionReceiver", DummyVisionReceiver)
    monkeypatch.setattr(runner_mod, "RefereeMessageReceiver", DummyRefereeReceiver)

    fake_runner = SimpleNamespace(
        mode=Mode.GRSIM,
        referee_system="official",
        start_threads=lambda vision_receiver, referee_receiver=None: started.append(
            (vision_receiver, referee_receiver)
        ),
    )

    StrategyRunner._setup_vision_and_referee(fake_runner)

    assert len(started) == 1
    assert isinstance(started[0][0], DummyVisionReceiver)
    assert isinstance(started[0][1], DummyRefereeReceiver)


def test_assert_exp_robots_valid(base_runner):
    base_runner._assert_exp_robots_and_ball(3, 3, True)  # Should not raise


def test_assert_exp_robots_too_many_friendly(base_runner):
    with pytest.raises(ValueError):
        base_runner._assert_exp_robots_and_ball(999, 3, True)


def test_assert_exp_robots_too_few_friendly(base_runner):
    with pytest.raises(ValueError):
        base_runner._assert_exp_robots_and_ball(0, 3, True)


def test_assert_exp_robots_too_many_enemy(base_runner):
    with pytest.raises(ValueError):
        base_runner._assert_exp_robots_and_ball(3, 999, True)


def test_assert_exp_goals_fails(base_runner):
    # Mock the strategy to return False on assert_exp_goals
    base_runner.my.strategy.assert_exp_goals = lambda *a, **k: False
    base_runner.my.game = MagicMock()
    base_runner.my.game.field = MagicMock(
        includes_my_goal_line=True,
        includes_opp_goal_line=True,
    )
    with pytest.raises(RuntimeError, match="Field does not match expected goals"):
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
    """Should raise ValueError when bounds are invalid."""
    invalid_bounds = FieldBounds(top_left=(3, 2), bottom_right=(-3, -2))

    with pytest.raises(ValueError):
        StrategyRunner(
            strategy=DummyStrategy(),
            my_team_is_yellow=True,
            my_team_is_right=True,
            mode="rsim",
            exp_friendly=3,
            exp_enemy=3,
            field_bounds=invalid_bounds,
        )


def test_strategy_runner_bounds_outside_non_standard_field_dims():
    """Should raise when bounds exceed a custom full field size."""
    # Valid in standard SSL dimensions, but outside GREAT_EXHIBITION_FIELD_DIMS.
    too_large_for_custom_dims = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))

    with pytest.raises(ValueError, match="out of full field bounds"):
        StrategyRunner(
            strategy=DummyStrategy(),
            my_team_is_yellow=True,
            my_team_is_right=True,
            mode="rsim",
            exp_friendly=3,
            exp_enemy=3,
            full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
            field_bounds=too_large_for_custom_dims,
        )
