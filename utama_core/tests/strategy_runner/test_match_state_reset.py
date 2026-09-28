"""A round-robin worker process plays many matches: a new `StrategyRunner` (one match)
must not inherit the last match's per-robot hysteresis."""

from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.shared import pass_and_score_geometry
from utama_core.skills.src import shielding
from utama_core.strategy.kernel_strategy import build_default_kernel_strategy
from utama_core.tests.shared.test_pass_and_score_geometry import _visual_game
from utama_core.tests.skills.test_shielding import _shielding


def test_a_new_runner_starts_with_no_robot_holding_or_committed():
    assert pass_and_score_geometry.has_ball(_visual_game((0.0, 0.0), 0.0, (0.10, 0.0)), 1, visual=True)
    assert _shielding(shielding.COMMIT_RANGE - 0.005) is False  # committed

    runner = StrategyRunner(
        strategy=AbstractStrategy(build_kernel_strategy=build_default_kernel_strategy((1, 2, 3, 4, 5))),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        exp_ball=True,
    )
    runner.close()

    mid = (pass_and_score_geometry._ACQUIRE_LATERAL_MAX + pass_and_score_geometry._RELEASE_LATERAL_MAX) / 2
    assert not pass_and_score_geometry.has_ball(_visual_game((0.0, 0.0), 0.0, (0.10, mid)), 1, visual=True)
    assert _shielding(shielding.COMMIT_RANGE + 0.01) is True
