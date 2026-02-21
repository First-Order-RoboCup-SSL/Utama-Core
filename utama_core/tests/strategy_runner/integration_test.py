from utama_core.entities.game.field import FieldBounds
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy


def test_position_refiner_config():
    runner = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=0,
        field_bounds=FieldBounds(top_left=(0, 3), bottom_right=(4.5, -3)),
    )

    assert runner.my.game.field.half_length == 2.25
    assert runner.my.game.field.half_width == 3.0
