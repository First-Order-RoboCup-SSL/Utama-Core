"""Integration test for `LeadAndSupportTactic` — drives it via a real
`StrategyRunner`/rsim `Game`, following `test_defense_tactic.py`'s pattern.
"""

from __future__ import annotations

import pytest

from utama_core.kernel.context import KernelContext
from utama_core.tactics.lead_and_support import LeadAndSupportTactic


@pytest.fixture
def runner():
    from utama_core.run.strategy_runner import StrategyRunner
    from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy

    r = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=5,
        exp_enemy=2,
        exp_ball=True,
    )
    yield r
    r.close()


@pytest.fixture
def game(runner):
    return runner.my.game


def _ctx(runner) -> KernelContext:
    motion_controller = runner.my.motion_controller(runner.mode, runner.rsim_env)
    return KernelContext(motion_controller=motion_controller, rsim_env=runner.rsim_env)


def test_produces_a_command_for_every_assigned_robot(game, runner):
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(runner), (1, 2, 3, 4), mem)
    assert set(commands.keys()) == {1, 2, 3, 4}


def test_single_robot_has_no_supports_but_still_gets_a_command(game, runner):
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(runner), (1,), mem)
    assert set(commands.keys()) == {1}
    assert mem.leader_id == 1


def test_leader_is_closest_robot_to_ball(game, runner):
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()
    _commands, mem = tactic.tick(game, _ctx(runner), (1, 2, 3, 4), mem)
    assert mem.leader_id in (1, 2, 3, 4)


def test_not_committed_before_leader_has_ball(game, runner):
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()
    assert tactic.is_committed(game, mem) is False
