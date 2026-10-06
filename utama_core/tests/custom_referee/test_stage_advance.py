"""The stage leaves NORMAL_FIRST_HALF_PRE when play starts, whether a person or the
referee's own auto-advance starts it. Only the manual path did, so a match run by
CustomReferee stayed "PRE" throughout: stage_time_left never counted down a playing
half, and score-aware strategies (strategy.pickers.is_late_in_half) never saw one.
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


# --- The match clock (rulebook, Game Stages) ---------------------------------------------
# "The match timer is paused whenever no team is allowed to manipulate the ball. This includes
# stop, halt and the preparation states of kick-off and penalty kick. Additionally, it is
# paused during ball placement." It used to be sim time since the stage began, so it ran
# through every stoppage, and nothing ended a half: it stopped at 0 for the rest of the match.


def _run(sm: GameStateMachine, t0: float, t1: float, dt: float = 1.0):
    """Step once a second over (t0, t1]; return the last RefereeData."""
    data = None
    t = t0
    while t < t1 - 1e-9:
        t += dt
        data = sm.step(t, None, _frame(t))
    return data


def _playing(sm: GameStateMachine, t: float) -> None:
    """Start (or resume) live play at `t`."""
    sm.force_command(RefereeCommand.FORCE_START, t)
    sm.step(t, None, _frame(t))


def test_the_match_clock_stops_whenever_no_team_may_play_the_ball():
    sm = _machine()
    _playing(sm, 10.0)
    _run(sm, 10.0, 20.0)  # 10 s of play
    for command, start in (
        (RefereeCommand.STOP, 20.0),
        (RefereeCommand.HALT, 30.0),
        (RefereeCommand.PREPARE_KICKOFF_BLUE, 40.0),
        (RefereeCommand.PREPARE_PENALTY_YELLOW, 50.0),
    ):
        sm.force_command(command, start)
        data = _run(sm, start, start + 10.0)
        assert data.stage_time_left == 290.0, command
    sm.force_command(RefereeCommand.BALL_PLACEMENT_YELLOW, 60.0, ball_placement_target=(1.0, 0.5))
    sm.step(60.0, None, _frame(60.0))
    assert sm.command == RefereeCommand.BALL_PLACEMENT_YELLOW
    _playing(sm, 60.0)
    assert _run(sm, 60.0, 65.0).stage_time_left == 285.0


def test_the_match_clock_runs_during_a_free_kick():
    sm = _machine()
    _playing(sm, 10.0)
    sm.force_command(RefereeCommand.DIRECT_FREE_BLUE, 20.0)
    assert _run(sm, 20.0, 25.0).stage_time_left == 285.0


def test_the_first_half_ends_after_its_playing_time_and_the_other_team_kicks_off_the_second():
    sm = _machine()  # profile kick-off team: yellow
    _playing(sm, 10.0)
    before = _run(sm, 10.0, 309.0)
    assert (before.stage, before.stage_time_left) == (Stage.NORMAL_FIRST_HALF, 1.0)

    after = sm.step(310.0, None, _frame(310.0))

    assert after.stage == Stage.NORMAL_SECOND_HALF_PRE
    assert after.stage_time_left == 300.0
    assert after.referee_command == RefereeCommand.STOP
    assert after.next_command == RefereeCommand.BALL_PLACEMENT_BLUE
    assert sm._post_ball_placement_command == RefereeCommand.PREPARE_KICKOFF_BLUE
    assert after.designated_position == (0.0, 0.0)


def test_the_second_half_starts_with_its_kick_off_and_ends_the_match():
    sm = _machine()
    _playing(sm, 10.0)
    _run(sm, 10.0, 310.0)
    sm.force_command(RefereeCommand.PREPARE_KICKOFF_BLUE, 320.0)
    sm.step(320.0, None, _frame(320.0))
    sm.force_command(RefereeCommand.NORMAL_START, 330.0)
    assert sm.step(330.0, None, _frame(330.0)).stage == Stage.NORMAL_SECOND_HALF

    assert _run(sm, 330.0, 629.0).stage == Stage.NORMAL_SECOND_HALF
    end = sm.step(630.0, None, _frame(630.0))

    assert (end.stage, end.referee_command, end.next_command) == (Stage.POST_GAME, RefereeCommand.STOP, None)
    assert end.designated_position is None  # else the sim runner would force a FORCE_START
    later = _run(sm, 630.0, 700.0)
    assert (later.stage, later.referee_command) == (Stage.POST_GAME, RefereeCommand.STOP)


def test_after_full_time_no_violation_restarts_play():
    from utama_core.custom_referee.rules.base_rule import RuleViolation

    sm = _machine()
    _playing(sm, 10.0)
    _run(sm, 10.0, 310.0)
    sm.force_command(RefereeCommand.FORCE_START, 320.0)
    _run(sm, 320.0, 621.0)
    assert sm.stage == Stage.POST_GAME
    goal = RuleViolation(
        rule_name="goal",
        suggested_command=RefereeCommand.STOP,
        next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
        status_message="Goal",
    )

    data = sm.step(700.0, goal, _frame(700.0))

    assert (data.referee_command, data.yellow_team.score) == (RefereeCommand.STOP, 0)


def test_an_operator_profile_ends_no_half_itself():
    from utama_core.custom_referee.profiles.profile_loader import AutoAdvanceConfig

    sm = GameStateMachine(
        half_duration_seconds=300.0,
        kickoff_team="yellow",
        n_robots_yellow=3,
        n_robots_blue=3,
        auto_advance=AutoAdvanceConfig(end_of_half=False),
    )
    sm.seed_clock(0.0)
    _playing(sm, 10.0)
    data = _run(sm, 10.0, 400.0)
    assert (data.stage, data.stage_time_left) == (Stage.NORMAL_FIRST_HALF, 0.0)


def test_late_in_half_means_the_last_minute_of_playing_time_not_of_sim_time():
    """strategy.pickers.is_late_in_half (score_aware_counter_flow, score_aware_zone_flow): with the
    clock counting sim time and stopping at 0, a 600 s round-robin match was "late" from 240 s
    of sim time to the end."""
    from types import SimpleNamespace

    from utama_core.strategy.pickers import is_late_in_half

    sm = _machine()
    _playing(sm, 0.0)
    _run(sm, 0.0, 100.0)  # 100 s played
    sm.force_command(RefereeCommand.STOP, 100.0)
    _run(sm, 100.0, 250.0)  # 150 s stopped
    _playing(sm, 250.0)
    at_330 = _run(sm, 250.0, 330.0)  # 180 s played in 330 s
    assert not is_late_in_half(SimpleNamespace(referee=at_330))
    at_391 = _run(sm, 330.0, 391.0)  # 241 s played
    assert is_late_in_half(SimpleNamespace(referee=at_391))
