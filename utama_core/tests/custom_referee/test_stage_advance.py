"""The stage leaves NORMAL_FIRST_HALF_PRE when play starts, whether a person or the
referee's own auto-advance starts it. Only the manual path did, so a match run by
CustomReferee stayed "PRE" throughout: stage_time_left never counted down a playing
half, and score-aware strategies (strategy.pickers.is_late_in_half) never saw one.
"""

from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
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


def test_the_teams_change_ends_at_half_time():
    """The referee says which team defends the +x goal (`blue_team_on_positive_half`, as the
    game controller does): taken from the first frame, swapped at half-time, and only then.
    `_frame` is yellow defending the left goal, so blue starts on the +x half."""
    sm = _machine()
    _playing(sm, 10.0)
    assert _run(sm, 10.0, 309.0).blue_team_on_positive_half is True

    assert sm.step(310.0, None, _frame(310.0)).blue_team_on_positive_half is False
    _playing(sm, 320.0)
    assert _run(sm, 320.0, 619.0).blue_team_on_positive_half is False
    assert sm.step(620.0, None, _frame(620.0)).stage == Stage.POST_GAME
    assert sm.step(621.0, None, _frame(621.0)).blue_team_on_positive_half is False


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


# --- Kick-off waits for every robot in its own half ---------------------------------------
# The kick-off started as soon as the kicker reached the centre circle. After the teams change
# ends every robot crosses the pitch, so the second half kicked off with robots still in the
# wrong half (found by test_change_of_ends on CI).


def _kickoff_frame(ts: float, straggler_x: float) -> GameFrame:
    """Yellow (friendly) defends the left goal and kicks off: its kicker in the circle, a
    second yellow robot at `straggler_x`; blue in its own (right) half."""
    zero = Vector3D(0, 0, 0)

    def robot(rid, x, friendly):
        return Robot(
            id=rid,
            is_friendly=friendly,
            has_ball=False,
            p=Vector2D(x, 0.0),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        )

    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={1: robot(1, -0.3, True), 2: robot(2, straggler_x, True)},
        enemy_robots={1: robot(1, 1.0, False)},
        ball=Ball(p=Vector3D(0.0, 0.0, 0), v=zero, a=zero),
        referee=None,
    )


def _kickoff_command_after(straggler_x: float, until: float) -> RefereeCommand:
    sm = _machine()
    sm.force_command(RefereeCommand.PREPARE_KICKOFF_YELLOW, 0.0)
    t, data = 0.0, None
    while t < until - 1e-9:
        t += 0.5
        data = sm.step(t, None, _kickoff_frame(t, straggler_x))
    return data.referee_command


def test_a_kickoff_waits_for_a_robot_still_in_the_other_half():
    assert _kickoff_command_after(straggler_x=-1.0, until=6.0) == RefereeCommand.NORMAL_START
    assert _kickoff_command_after(straggler_x=0.5, until=6.0) == RefereeCommand.PREPARE_KICKOFF_YELLOW


def test_a_robot_on_the_halfway_line_counts_as_in_its_own_half():
    assert _kickoff_command_after(straggler_x=0.08, until=6.0) == RefereeCommand.NORMAL_START


def test_a_robot_that_never_gets_back_holds_the_kickoff_up_only_so_long():
    # force_command counts the 3 s preparation as already done: the 10 s cap, then the 2 s sustain
    assert _kickoff_command_after(straggler_x=0.5, until=11.5) == RefereeCommand.PREPARE_KICKOFF_YELLOW
    assert _kickoff_command_after(straggler_x=0.5, until=12.0) == RefereeCommand.NORMAL_START
