"""Integration test: `KernelStrategy` driven by a real `StrategyRunner` in rsim.

Exercises the actual `StrategyRunner` call sequence (`load_robot_controller` /
`load_motion_controller` before `load_game`, then repeated `step_once()`)
against `KernelStrategy`, not just the adapter in isolation — this is the
thing `test_strategy.py`/`test_two_robot_attack_tactic.py` don't cover: that
the kernel `Strategy` actually gets wired up correctly through
`AbstractStrategy`'s blackboard/robot_controller plumbing.
"""

from __future__ import annotations

import pytest

from utama_core.kernel.kernel_strategy import (
    KernelStrategy,
    build_default_kernel_strategy,
)


@pytest.fixture
def kernel_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = KernelStrategy(build_kernel_strategy=build_default_kernel_strategy((1, 2)))
    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=0,
        exp_ball=True,
    )
    yield runner
    runner.close()


def test_kernel_strategy_builds_kernel_strategy_on_load_game(kernel_runner):
    strategy = kernel_runner.my.strategy
    assert strategy._kernel_strategy is not None


def test_kernel_strategy_steps_without_error(kernel_runner):
    for _ in range(5):
        kernel_runner.step_once()


def test_kernel_strategy_goalkeeper_is_pinned_to_robot_0(kernel_runner):
    strategy = kernel_runner.my.strategy
    assert strategy._goalkeeper_id == 0
    kernel_runner.step_once()
    # Goalkeeper's own tactic mem is tracked outside the kernel Strategy's slots.
    assert strategy._kernel_strategy.active_tactic_id != "goalkeeper"


def test_kernel_strategy_outfield_tactic_gets_ticked(kernel_runner):
    kernel_runner.step_once()
    strategy = kernel_runner.my.strategy
    assert strategy._kernel_strategy.active_tactic_id == "two_robot_attack"
