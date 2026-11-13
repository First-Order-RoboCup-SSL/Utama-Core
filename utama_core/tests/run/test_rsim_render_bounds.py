from unittest.mock import MagicMock, patch

from utama_core.config.enums import Mode
from utama_core.entities.game.field import FieldBounds
from utama_core.run.strategy_runner import StrategyRunner


def _make_runner_for_overlay_tests(render_mode: str | None):
    with patch.object(StrategyRunner, "__init__", lambda self: None):
        runner = StrategyRunner()
        runner.mode = Mode.RSIM
        runner.field_bounds = FieldBounds(top_left=(-1.0, 2.0), bottom_right=(3.0, -4.0))
        runner.rsim_env = MagicMock()
        runner.rsim_env.render_mode = render_mode
        return runner


def test_draw_rsim_field_bounds_overlay_draws_expected_polygon():
    runner = _make_runner_for_overlay_tests(render_mode="human")

    runner._draw_rsim_field_bounds_overlay()

    runner.rsim_env.draw_polygon.assert_called_once_with(
        [(-1.0, 2.0), (3.0, 2.0), (3.0, -4.0), (-1.0, -4.0)],
        color="PINK",
        width=2,
    )


def test_draw_rsim_field_bounds_overlay_not_drawn_when_not_human():
    runner = _make_runner_for_overlay_tests(render_mode=None)

    runner._draw_rsim_field_bounds_overlay()

    runner.rsim_env.draw_polygon.assert_not_called()
