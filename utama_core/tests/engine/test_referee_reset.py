"""Tests for the referee-command-transition -> reset-tier classification."""

from __future__ import annotations

from utama_core.engine.referee_override import is_override_command
from utama_core.engine.referee_reset import ResetTier, classify_transition, is_paused
from utama_core.entities.referee.referee_command import RefereeCommand

# GOAL_YELLOW/GOAL_BLUE are deprecated and never emitted by `CustomReferee`
# (which goes straight to STOP + next_command on a goal — see
# `GameStateMachine._handle_goal`), so they are legitimately unhandled by
# either mechanism below without that being a live bug — excluded here so the
# exhaustiveness test stays meaningful instead of failing on dead enum values.
_DEPRECATED_UNHANDLED_COMMANDS = frozenset({RefereeCommand.GOAL_YELLOW, RefereeCommand.GOAL_BLUE})


def test_no_transition_when_command_unchanged():
    assert classify_transition(RefereeCommand.NORMAL_START, RefereeCommand.NORMAL_START) is ResetTier.NONE


def test_entering_a_restart_command_is_a_barrier():
    assert classify_transition(RefereeCommand.NORMAL_START, RefereeCommand.PREPARE_KICKOFF_YELLOW) is ResetTier.BARRIER
    assert classify_transition(RefereeCommand.STOP, RefereeCommand.BALL_PLACEMENT_BLUE) is ResetTier.BARRIER
    assert classify_transition(RefereeCommand.STOP, RefereeCommand.DIRECT_FREE_YELLOW) is ResetTier.BARRIER


def test_resuming_from_a_barrier_command_is_still_a_barrier():
    assert classify_transition(RefereeCommand.PREPARE_KICKOFF_YELLOW, RefereeCommand.NORMAL_START) is ResetTier.BARRIER
    assert classify_transition(RefereeCommand.BALL_PLACEMENT_BLUE, RefereeCommand.FORCE_START) is ResetTier.BARRIER


def test_resuming_from_a_pause_with_normal_start_is_not_a_barrier():
    assert classify_transition(RefereeCommand.HALT, RefereeCommand.NORMAL_START) is ResetTier.NONE
    assert classify_transition(RefereeCommand.STOP, RefereeCommand.NORMAL_START) is ResetTier.NONE


def test_force_start_out_of_a_pause_is_a_barrier():
    """FORCE_START restarts with the ball wherever it now is (rsim may have teleported it
    during the stop), so pre-pause commitments must not survive it."""
    assert classify_transition(RefereeCommand.STOP, RefereeCommand.FORCE_START) is ResetTier.BARRIER
    assert classify_transition(RefereeCommand.HALT, RefereeCommand.FORCE_START) is ResetTier.BARRIER


def test_entering_a_pause_command_is_a_pause_not_a_barrier():
    assert classify_transition(RefereeCommand.NORMAL_START, RefereeCommand.STOP) is ResetTier.PAUSE
    assert classify_transition(RefereeCommand.FORCE_START, RefereeCommand.HALT) is ResetTier.PAUSE


def test_first_tick_with_no_previous_command_is_not_a_barrier_unless_already_in_one():
    assert classify_transition(None, RefereeCommand.NORMAL_START) is ResetTier.NONE
    assert classify_transition(None, RefereeCommand.PREPARE_KICKOFF_YELLOW) is ResetTier.BARRIER


def test_is_paused():
    assert is_paused(RefereeCommand.STOP) is True
    assert is_paused(RefereeCommand.HALT) is True
    assert is_paused(RefereeCommand.NORMAL_START) is False
    assert is_paused(RefereeCommand.BALL_PLACEMENT_YELLOW) is False


def test_every_referee_command_is_routed_somewhere():
    """Every live `RefereeCommand` must be handled by exactly one of:
    `is_paused` (freeze), `is_override_command` (legal-position takeover), or
    the live-play fallthrough (`NORMAL_START`/`FORCE_START`, where tactics
    tick normally).

    Regression test for a real, previously-shipped gap: `TIMEOUT_YELLOW`/
    `TIMEOUT_BLUE` were in neither set, so a timeout left tactics ticking and
    issuing ordinary motion commands straight through it. The two
    hand-maintained frozensets (`_PAUSE_COMMANDS`, `_OVERRIDE_COMMANDS`) can
    silently drift out of sync with `RefereeCommand` as it changes — this
    test is the mechanical check against that.
    """
    for command in RefereeCommand:
        if command in _DEPRECATED_UNHANDLED_COMMANDS:
            continue
        handled = (
            is_paused(command)
            or is_override_command(command)
            or command in (RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START)
        )
        assert handled, f"{command.name} is not routed to a pause, an override, or live play"
