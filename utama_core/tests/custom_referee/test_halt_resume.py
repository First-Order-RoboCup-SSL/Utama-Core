"""A HALT resumes into the restart it interrupted, not into open play.

DefenseAreaStoppageRule escalates a second foul to HALT with no restart of its own. The sim
resumed a HALT with `force_command(STOP)`, which cleared the queued restart, so play went on
as FORCE_START: on CI the second half's kick-off was lost this way (half-time STOP with a queued
BALL_PLACEMENT_BLUE -> PREPARE_KICKOFF_BLUE, HALT at 12 s, FORCE_START at 21 s). A HALT arriving
during the restart itself (a kick-off being prepared, a ball being placed) lost it the same way.
"""

from __future__ import annotations

from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.tests.custom_referee.helpers import ball, frame

_HALT = RuleViolation(
    rule_name="defense_area_stoppage",
    suggested_command=RefereeCommand.HALT,
    next_command=None,
    status_message="2nd foul — HALT",
    offending_teams=(True,),
)


def _machine() -> GameStateMachine:
    sm = GameStateMachine(half_duration_seconds=300.0, kickoff_team="yellow", n_robots_yellow=3, n_robots_blue=3)
    sm.seed_clock(0.0)
    return sm


def _clear(ts: float):
    """No robot near the ball, so STOP may advance to its queued restart."""
    return frame(ball(0.0, 0.0), ts=ts)


def _halt_and_resume(sm: GameStateMachine, t: float) -> bool:
    sm.step(t, _HALT, _clear(t))
    assert sm.command == RefereeCommand.HALT
    return sm.resume_from_halt(t + 5.0)


def test_a_restart_queued_behind_a_stop_survives_a_halt():
    sm = _machine()
    sm.force_command(RefereeCommand.STOP, 1.0)
    sm.next_command = RefereeCommand.BALL_PLACEMENT_BLUE
    sm._post_ball_placement_command = RefereeCommand.PREPARE_KICKOFF_BLUE

    assert _halt_and_resume(sm, 10.0) is True
    assert (sm.command, sm.next_command) == (RefereeCommand.STOP, RefereeCommand.BALL_PLACEMENT_BLUE)

    sm.step(15.1, None, _clear(15.1))
    assert (sm.command, sm.next_command) == (RefereeCommand.BALL_PLACEMENT_BLUE, RefereeCommand.PREPARE_KICKOFF_BLUE)


def test_a_kickoff_being_prepared_is_prepared_again_after_a_halt():
    sm = _machine()
    sm.force_command(RefereeCommand.PREPARE_KICKOFF_BLUE, 1.0)

    assert _halt_and_resume(sm, 10.0) is True
    sm.step(15.1, None, _clear(15.1))
    assert sm.command == RefereeCommand.PREPARE_KICKOFF_BLUE


def test_a_ball_placement_in_progress_continues_after_a_halt():
    sm = _machine()
    sm.force_command(RefereeCommand.BALL_PLACEMENT_YELLOW, 1.0, ball_placement_target=(1.0, 0.5))
    sm.next_command = RefereeCommand.DIRECT_FREE_YELLOW

    assert _halt_and_resume(sm, 10.0) is True
    sm.step(15.1, None, _clear(15.1))
    assert (sm.command, sm.next_command) == (RefereeCommand.BALL_PLACEMENT_YELLOW, RefereeCommand.DIRECT_FREE_YELLOW)


def test_a_halt_in_open_play_has_nothing_to_resume_into():
    sm = _machine()
    sm.force_command(RefereeCommand.FORCE_START, 1.0)

    assert _halt_and_resume(sm, 10.0) is False
    assert sm.command == RefereeCommand.STOP
