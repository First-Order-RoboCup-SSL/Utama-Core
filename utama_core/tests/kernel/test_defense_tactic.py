"""Integration test for `DefenseTactic` — drives it via a real `StrategyRunner`/rsim
`Game`, since `defend_parameter` reads real field/ball geometry that isn't worth
faking with a stand-in object (unlike the pure-logic kernel/scheduler tests).
"""

from __future__ import annotations

import pytest

from utama_core.kernel.context import KernelContext
from utama_core.tactics.defense import DefenseTactic


@pytest.fixture
def runner():
    from utama_core.run.strategy_runner import StrategyRunner
    from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy

    r = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=0,
        exp_ball=True,
    )
    yield r
    r.close()


@pytest.fixture
def game(runner):
    return runner.my.game


def _ctx(runner) -> KernelContext:
    motion_controller = runner.my.motion_controller(runner.mode, runner.rsim_env)
    return KernelContext(motion_controller=motion_controller)


def test_defense_tactic_produces_a_command_for_each_assigned_robot(game, runner):
    tactic = DefenseTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(runner), (1, 2), mem)
    assert set(commands.keys()) == {1, 2}


def test_defense_tactic_single_defender(game, runner):
    tactic = DefenseTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(runner), (1,), mem)
    assert set(commands.keys()) == {1}


def test_defense_tactic_never_commits(game):
    """No phase state — always freely reassignable, unlike two_robot_attack."""
    tactic = DefenseTactic()
    mem = tactic.initial_mem()
    assert tactic.is_committed(game, mem) is False
