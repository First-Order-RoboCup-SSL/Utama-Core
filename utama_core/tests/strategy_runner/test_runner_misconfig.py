from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from utama_core.config.enums import Mode
from utama_core.config.field_params import (
    GREAT_EXHIBITION_FIELD_DIMS,
    STANDARD_FIELD_DIMS,
)
from utama_core.custom_referee import CustomReferee
from utama_core.entities.game.field import FieldBounds
from utama_core.run.referee_source import OfficialReferee
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


def test_validate_referee_defaults_to_none_in_rsim():
    assert StrategyRunner._validate_referee(Mode.RSIM, None) is None


def test_validate_referee_defaults_to_none_in_grsim():
    assert StrategyRunner._validate_referee(Mode.GRSIM, None) is None


def test_validate_referee_rejects_unknown_type():
    with pytest.raises(TypeError, match="OfficialReferee"):
        StrategyRunner._validate_referee(Mode.RSIM, object())


def test_validate_referee_rejects_official_in_rsim():
    with pytest.raises(ValueError, match="OfficialReferee"):
        StrategyRunner._validate_referee(Mode.RSIM, OfficialReferee())


def test_validate_referee_accepts_official_in_grsim():
    result = StrategyRunner._validate_referee(Mode.GRSIM, OfficialReferee())
    assert isinstance(result, OfficialReferee)


def _make_fake_field_size(dims=STANDARD_FIELD_DIMS):
    """Build a fake SSL_GeometryFieldSize-like object from a FieldDimensions."""
    return SimpleNamespace(
        field_length=int(dims.full_field_half_length * 2000),
        field_width=int(dims.full_field_half_width * 2000),
        goal_width=int(dims.half_goal_width * 2000),
        penalty_area_depth=int(dims.half_defense_area_depth * 2000),
        penalty_area_width=int(dims.half_defense_area_width * 2000),
        field_arcs=[SimpleNamespace(name="CenterCircle", radius=int(dims.center_circle_radius * 1000))],
    )


def test_setup_vision_and_referee_starts_vision_only_when_referee_none(monkeypatch):
    from utama_core.run import strategy_runner as runner_mod

    started = []

    class DummyVisionReceiver:
        def __init__(self, buffers, on_geometry=None):
            pass

    class DummyRefereeReceiver:
        def __init__(self, buffer):
            pass

    monkeypatch.setattr(runner_mod, "VisionReceiver", DummyVisionReceiver)
    monkeypatch.setattr(runner_mod, "RefereeMessageReceiver", DummyRefereeReceiver)

    fake_runner = SimpleNamespace(
        mode=Mode.GRSIM,
        referee=None,
        full_field_dims=STANDARD_FIELD_DIMS,
        start_threads=lambda *args: started.append(args),
        _make_geometry_validation_callback=lambda: None,
    )

    vision_buffers, ref_buffer = StrategyRunner._setup_vision_and_referee(fake_runner)

    assert len(vision_buffers) > 0
    assert ref_buffer.maxlen == 1
    assert len(started) == 1
    assert isinstance(started[0][0], DummyVisionReceiver)
    assert len(started[0]) == 1


def test_setup_vision_and_referee_starts_vision_only_when_referee_custom(monkeypatch):
    from utama_core.run import strategy_runner as runner_mod

    started = []

    class DummyVisionReceiver:
        def __init__(self, buffers, on_geometry=None):
            pass

    monkeypatch.setattr(runner_mod, "VisionReceiver", DummyVisionReceiver)

    fake_runner = SimpleNamespace(
        mode=Mode.REAL,
        referee=MagicMock(spec=CustomReferee),
        full_field_dims=STANDARD_FIELD_DIMS,
        start_threads=lambda *args: started.append(args),
        _make_geometry_validation_callback=lambda: None,
    )

    StrategyRunner._setup_vision_and_referee(fake_runner)

    assert len(started) == 1
    assert len(started[0]) == 1


def test_setup_vision_and_referee_starts_both_receivers_when_referee_official(
    monkeypatch,
):
    from utama_core.run import strategy_runner as runner_mod

    started = []

    class DummyVisionReceiver:
        def __init__(self, buffers, on_geometry=None):
            pass

    class DummyRefereeReceiver:
        def __init__(self, buffer):
            pass

    monkeypatch.setattr(runner_mod, "VisionReceiver", DummyVisionReceiver)
    monkeypatch.setattr(runner_mod, "RefereeMessageReceiver", DummyRefereeReceiver)

    fake_runner = SimpleNamespace(
        mode=Mode.GRSIM,
        referee=OfficialReferee(),
        full_field_dims=STANDARD_FIELD_DIMS,
        start_threads=lambda *args: started.append(args),
        _make_geometry_validation_callback=lambda: None,
    )

    StrategyRunner._setup_vision_and_referee(fake_runner)

    assert len(started) == 1
    assert isinstance(started[0][0], DummyVisionReceiver)
    assert isinstance(started[0][1], DummyRefereeReceiver)


def test_geometry_validation_passes_on_matching_field(monkeypatch):
    fake_field_size = _make_fake_field_size(STANDARD_FIELD_DIMS)
    fake_runner = SimpleNamespace(full_field_dims=STANDARD_FIELD_DIMS)
    callback = StrategyRunner._make_geometry_validation_callback(fake_runner)
    callback(fake_field_size)  # should not raise


def test_geometry_validation_raises_on_mismatched_field(monkeypatch):
    fake_field_size = _make_fake_field_size(GREAT_EXHIBITION_FIELD_DIMS)
    fake_runner = SimpleNamespace(full_field_dims=STANDARD_FIELD_DIMS)
    callback = StrategyRunner._make_geometry_validation_callback(fake_runner)
    with pytest.raises(RuntimeError, match="Field geometry mismatch"):
        callback(fake_field_size)


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


def test_validate_vision_to_cmd_mapping_real_not_controlled():
    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=None,
        exp_friendly=3,
    )
    result = StrategyRunner._validate_vision_to_cmd_mapping(runner, None, True)
    assert result == {}


def test_validate_vision_to_cmd_mapping_pvp_opp_mapping_missing():
    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=True,
        exp_friendly=3,
        exp_enemy=3,
    )
    with pytest.raises(ValueError, match="required for both teams in real PVP"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, None, False)


def test_validate_vision_to_cmd_mapping_pvp_friendly_mapping_missing():
    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=True,
        exp_friendly=3,
        exp_enemy=3,
    )
    with pytest.raises(ValueError, match="required for both teams in real PVP"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, None, True)


def test_validate_vision_to_cmd_mapping_type_error():
    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=None,
        exp_friendly=3,
    )
    with pytest.raises(TypeError, match="must be a dictionary"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, [], True)


def test_validate_vision_to_cmd_mapping_ignored_warning():
    import warnings

    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=None,
        exp_friendly=3,
    )
    with pytest.warns(UserWarning, match="vision_to_cmd_mapping is provided but will be ignored"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {0: 0}, False)


def test_validate_vision_to_cmd_mapping_incorrect_length_friendly():
    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=True,
        exp_friendly=3,
        exp_enemy=3,
    )
    # 2 entries when 3 are expected
    with pytest.raises(ValueError, match="has 2 entries but 3 robots are expected"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {0: 0, 1: 1}, True)


def test_validate_vision_to_cmd_mapping_incorrect_length_enemy():
    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=True,
        exp_friendly=3,
        exp_enemy=3,
    )
    # 2 entries when 3 are expected
    with pytest.raises(ValueError, match="has 2 entries but 3 robots are expected"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {0: 0, 1: 1}, False)


def test_validate_vision_to_cmd_mapping_correct_count_non_contiguous_ids_passes_init():
    # Non-contiguous vision IDs (e.g. real field robots numbered 5,6,7) with the right
    # count must PASS at init time — coverage against observed IDs is validated later
    # in _validate_mapping_covers_game_frame() after _load_game().
    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=True,
        exp_friendly=3,
        exp_enemy=3,
    )
    result = StrategyRunner._validate_vision_to_cmd_mapping(runner, {5: 0, 6: 1, 7: 2}, True)
    assert result == {5: 0, 6: 1, 7: 2}


def test_validate_mapping_covers_game_frame_mismatch_raises():
    from utama_core.run.strategy_runner import StrategyRunner

    runner = SimpleNamespace()
    with pytest.raises(ValueError, match="missing entries for observed IDs"):
        StrategyRunner._validate_mapping_covers_game_frame(runner, {0: 0, 1: 1, 2: 2}, {5, 6, 7}, "friendly")


def test_validate_mapping_covers_game_frame_match_passes():
    from utama_core.run.strategy_runner import StrategyRunner

    runner = SimpleNamespace()
    # Should not raise
    StrategyRunner._validate_mapping_covers_game_frame(runner, {5: 0, 6: 1, 7: 2}, {5, 6, 7}, "friendly")


def test_validate_mapping_covers_game_frame_empty_mapping_passes():
    from utama_core.run.strategy_runner import StrategyRunner

    runner = SimpleNamespace()
    # Empty mapping (non-PVP mode) always passes
    StrategyRunner._validate_mapping_covers_game_frame(runner, {}, {0, 1, 2}, "friendly")


def test_validate_vision_to_cmd_mapping_invalid_ids():
    from utama_core.config.physical_constants import MAX_ROBOT_ID

    runner = SimpleNamespace(
        mode=Mode.REAL,
        my_team_is_yellow=True,
        opp=None,
        exp_friendly=3,
    )
    with pytest.raises(TypeError, match="must map integers to integers"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {0: "0"}, True)
    with pytest.raises(ValueError, match="cannot have negative IDs"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {-1: 0}, True)
    with pytest.raises(ValueError, match="cannot have vision IDs greater than"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {MAX_ROBOT_ID + 1: 0}, True)
    with pytest.raises(ValueError, match="cannot have command IDs greater than 255"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {0: 256}, True)


def test_validate_vision_to_cmd_mapping_sim_mode_raises():
    runner = SimpleNamespace(
        mode=Mode.RSIM,
        my_team_is_yellow=True,
        opp=None,
    )
    with pytest.raises(ValueError, match="should not be provided in simulation modes"):
        StrategyRunner._validate_vision_to_cmd_mapping(runner, {0: 0}, True)


def test_trusted_ir_robots_yellow_team_is_my_team():
    """yellow_trusted_ir_robots routes to my refiner when my_team_is_yellow=True."""
    runner = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        yellow_trusted_ir_robots=frozenset({0, 1}),
    )
    assert runner.my.robot_info_refiner._trusted_ir_robots == frozenset({0, 1})
    assert runner.opp is None


def test_trusted_ir_robots_blue_team_is_my_team():
    """blue_trusted_ir_robots routes to my refiner when my_team_is_yellow=False."""
    runner = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=False,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        blue_trusted_ir_robots=frozenset({2}),
    )
    assert runner.my.robot_info_refiner._trusted_ir_robots == frozenset({2})


def test_trusted_ir_robots_both_teams_pvp():
    """Both colour params route correctly in PVP mode."""
    runner = StrategyRunner(
        strategy=DummyStrategy(),
        opp_strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        yellow_trusted_ir_robots=frozenset({0}),
        blue_trusted_ir_robots=frozenset({1}),
    )
    assert runner.my.robot_info_refiner._trusted_ir_robots == frozenset({0})
    assert runner.opp.robot_info_refiner._trusted_ir_robots == frozenset({1})


def test_trusted_ir_robots_none_by_default():
    """Default None means trust all IR sensors (backwards-compatible)."""
    runner = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
    )
    assert runner.my.robot_info_refiner._trusted_ir_robots is None


def test_check_no_cmd_duplicate_if_transmission_sharing():
    runner = SimpleNamespace()

    # Test valid non-overlapping mappings
    yellow_mapping = {0: 1, 1: 2}
    blue_mapping = {0: 3, 1: 4}
    StrategyRunner._check_no_cmd_duplicate_if_transmission_sharing(runner, yellow_mapping, blue_mapping)

    # Test collision in same team
    yellow_mapping_duplicate = {0: 1, 1: 1}
    with pytest.raises(ValueError, match="cannot have overlapping command IDs"):
        StrategyRunner._check_no_cmd_duplicate_if_transmission_sharing(runner, yellow_mapping_duplicate, blue_mapping)

    # Test collision across teams
    yellow_mapping = {0: 1, 1: 2}
    blue_mapping_overlap = {0: 2, 1: 4}
    with pytest.raises(ValueError, match="cannot have overlapping command IDs"):
        StrategyRunner._check_no_cmd_duplicate_if_transmission_sharing(runner, yellow_mapping, blue_mapping_overlap)
