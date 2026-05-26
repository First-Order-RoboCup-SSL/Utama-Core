"""Focused unit tests for CustomReferee GameStateMachine foul routing."""

from __future__ import annotations

from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.entities.referee.referee_command import RefereeCommand


def _state_machine() -> GameStateMachine:
    sm = GameStateMachine(
        half_duration_seconds=300.0,
        kickoff_team="yellow",
        n_robots_yellow=3,
        n_robots_blue=3,
    )
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.NORMAL_START, 0.0)
    return sm


def test_handle_foul_with_designated_position_routes_through_ball_placement():
    sm = _state_machine()
    violation = RuleViolation(
        rule_name="out_of_bounds",
        suggested_command=RefereeCommand.STOP,
        next_command=RefereeCommand.DIRECT_FREE_BLUE,
        status_message="Ball out of bounds",
        designated_position=(1.2, -0.8),
    )

    data = sm.step(current_time=10.0, violation=violation)

    assert data.referee_command == RefereeCommand.STOP
    assert data.next_command == RefereeCommand.BALL_PLACEMENT_BLUE
    assert sm._post_ball_placement_command == RefereeCommand.DIRECT_FREE_BLUE
    assert data.designated_position == (1.2, -0.8)


def test_handle_foul_without_designated_position_routes_directly_to_restart():
    sm = _state_machine()
    violation = RuleViolation(
        rule_name="defense_area",
        suggested_command=RefereeCommand.STOP,
        next_command=RefereeCommand.DIRECT_FREE_BLUE,
        status_message="Defense area violation",
        designated_position=None,
    )

    data = sm.step(current_time=10.0, violation=violation)

    assert data.referee_command == RefereeCommand.STOP
    assert data.next_command == RefereeCommand.DIRECT_FREE_BLUE
    assert sm._post_ball_placement_command is None
    assert data.designated_position is None
