"""The stage leaves NORMAL_FIRST_HALF_PRE when play starts, whether a person or the
referee's own auto-advance starts it. Only the manual path did, so a match run by
CustomReferee stayed "PRE" throughout: stage_time_left never counted down a playing
half, and score-aware strategies (strategy.pickers._is_late_in_half) never saw one.
"""

from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.tests.custom_referee.test_free_kick_timeout import _frame


def _machine() -> GameStateMachine:
    sm = GameStateMachine(half_duration_seconds=300.0, kickoff_team="yellow", n_robots_yellow=3, n_robots_blue=3)
    sm.seed_clock(0.0)
    return sm


def test_play_started_by_the_referee_s_auto_advance_starts_the_half():
    sm = _machine()
    sm.force_command(RefereeCommand.DIRECT_FREE_YELLOW, 0.0)  # force: set_command would already advance

    assert sm.step(9.9, None, _frame(9.9)).stage == Stage.NORMAL_FIRST_HALF_PRE
    after = sm.step(10.0, None, _frame(10.0))  # untaken free kick: FORCE_START

    assert after.referee_command == RefereeCommand.FORCE_START
    assert after.stage == Stage.NORMAL_FIRST_HALF
    assert after.stage_time_left == 300.0  # the half's clock starts when play does


def test_a_restart_during_the_half_does_not_restart_its_clock():
    sm = _machine()
    sm.force_command(RefereeCommand.DIRECT_FREE_YELLOW, 0.0)
    sm.step(10.0, None, _frame(10.0))
    sm.force_command(RefereeCommand.DIRECT_FREE_BLUE, 50.0)

    after = sm.step(60.0, None, _frame(60.0))

    assert after.referee_command == RefereeCommand.FORCE_START
    assert after.stage == Stage.NORMAL_FIRST_HALF
    assert after.stage_time_left == 250.0
