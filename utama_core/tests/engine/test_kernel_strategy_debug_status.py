"""Tests for `AbstractStrategy.debug_status()` / `Strategy.slot_status()` — the
Tactic model's replacement for `_push_bt_nodes_to_referee`'s BT-node walk,
used to show "what is this robot's tactic doing right now" in the referee
GUI when there is no behaviour tree to walk.
"""

from __future__ import annotations

import pytest

from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.strategy.kernel_strategy import build_split_shape_kernel_strategy


@pytest.fixture
def split_shape_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = AbstractStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))
    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        exp_ball=True,
    )
    yield runner
    runner.close()


def test_debug_status_covers_goalkeeper_and_all_outfield_robots(split_shape_runner):
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    status = strategy.debug_status()

    assert status[0] == ["goalkeeper"]
    for robot_id in (1, 2, 3, 4, 5):
        assert robot_id in status
        assert len(status[robot_id]) == 1


def test_debug_status_labels_match_active_partition(split_shape_runner):
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    status = strategy.debug_status()
    partition = strategy._kernel_strategy.active_partition

    for tactic_id, robots in partition.items():
        for robot_id in robots:
            assert status[robot_id][0].startswith(tactic_id)


def test_slot_status_reports_committed_flag(split_shape_runner):
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    game = split_shape_runner.my.game
    status = strategy._kernel_strategy.slot_status(game)

    for tactic_id, info in status.items():
        assert isinstance(info["committed"], bool)
        assert info["robots"]
