"""`StrategyRunner.replace_strategy`: the dashboard's Strategy panel changes, or restarts, what a
side plays while the run carries on — only while HALTed, so no robot is mid-move."""

from __future__ import annotations

import pytest

from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import TickContext
from utama_core.engine.match_log import MatchLog
from utama_core.engine.strategy import Strategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.tactics.defense import DefenseTactic


def _strategy(ticks: list) -> AbstractStrategy:
    def build(motion_controller) -> Strategy:
        kernel = Strategy(
            tactics={"play": DefenseTactic()},
            partitioner=Strategy.single_tactic_picker(lambda game, active: "play"),
            outfield_robot_ids=(1,),
            ctx=TickContext(motion_controller=motion_controller),
        )
        tick = kernel.tick
        kernel.tick = lambda game: ticks.append(game.referee.referee_command) or tick(game)
        return kernel

    return AbstractStrategy(build_kernel_strategy=build)


@pytest.fixture
def runner():
    referee = CustomReferee(load_profile("simulation"), n_robots_yellow=2, n_robots_blue=0)
    r = StrategyRunner(
        strategy=_strategy([]),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=2,
        exp_enemy=0,
        referee=referee,
        referee_initial_command=RefereeCommand.HALT,
    )
    r.match_log = MatchLog()
    yield r
    r.close()


def test_a_replacement_runs_from_the_next_tick_and_logs_to_the_match_log(runner):
    runner._run_step()
    new_ticks: list = []
    runner.replace_strategy("my", _strategy(new_ticks))
    assert not new_ticks
    runner._run_step()
    assert new_ticks == [RefereeCommand.HALT]
    assert runner.my.strategy._kernel_strategy.match_log is runner.match_log


def test_a_replacement_is_refused_unless_halted(runner):
    runner._run_step()
    runner.referee.set_command(RefereeCommand.FORCE_START, runner.my.current_game_frame.ts)
    runner._run_step()
    with pytest.raises(ValueError, match="Halt"):
        runner.replace_strategy("my", _strategy([]))


def test_a_replacement_queued_while_halted_is_dropped_if_play_starts_first(runner):
    runner._run_step()
    old = runner.my.strategy
    runner.replace_strategy("my", _strategy([]))
    runner.referee.set_command(RefereeCommand.FORCE_START, runner.my.current_game_frame.ts)
    runner._run_step()
    assert runner.my.strategy is old


def test_there_is_no_opp_side_to_replace_without_an_opponent(runner):
    runner._run_step()
    with pytest.raises(ValueError, match="opp"):
        runner.replace_strategy("opp", _strategy([]))
