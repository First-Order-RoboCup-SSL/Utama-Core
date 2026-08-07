"""End-to-end test: `KernelStrategy` wired with `build_split_shape_kernel_strategy`,
driven by a real `StrategyRunner` with 6 robots (5 outfield + pinned goalkeeper),
exercising the full split-shape path: `Strategy` picking a concurrent
LeadAndSupportTactic/ShadowAndMarkTactic split, ticked through the real runner.
"""

from __future__ import annotations

import pytest

from utama_core.kernel.kernel_strategy import (
    KernelStrategy,
    build_split_shape_kernel_strategy,
)
from utama_core.kernel.strategy import Strategy


@pytest.fixture
def split_shape_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = KernelStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))
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


def test_kernel_strategy_builds_a_strategy(split_shape_runner):
    strategy = split_shape_runner.my.strategy
    assert isinstance(strategy._kernel_strategy, Strategy)


def test_steps_without_error_for_several_ticks(split_shape_runner):
    for _ in range(10):
        split_shape_runner.step_once()


def test_partition_covers_all_five_outfield_robots_every_tick(split_shape_runner):
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    partition = strategy._kernel_strategy.active_partition
    all_assigned = frozenset().union(*partition.values()) if partition else frozenset()
    assert all_assigned == frozenset({1, 2, 3, 4, 5})


def test_goalkeeper_still_pinned_to_robot_0(split_shape_runner):
    strategy = split_shape_runner.my.strategy
    assert strategy._goalkeeper_id == 0
    split_shape_runner.step_once()
    partition = strategy._kernel_strategy.active_partition
    for robots in partition.values():
        assert 0 not in robots
