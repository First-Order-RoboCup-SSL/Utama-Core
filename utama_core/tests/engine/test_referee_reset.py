"""Tests for the referee-command-transition -> reset-tier classification."""

from __future__ import annotations

from utama_core.engine.referee_reset import ResetTier, classify_transition, is_paused
from utama_core.entities.referee.referee_command import RefereeCommand


def test_no_transition_when_command_unchanged():
    assert classify_transition(RefereeCommand.NORMAL_START, RefereeCommand.NORMAL_START) is ResetTier.NONE


def test_entering_a_restart_command_is_a_barrier():
    assert classify_transition(RefereeCommand.NORMAL_START, RefereeCommand.PREPARE_KICKOFF_YELLOW) is ResetTier.BARRIER
    assert classify_transition(RefereeCommand.STOP, RefereeCommand.BALL_PLACEMENT_BLUE) is ResetTier.BARRIER
    assert classify_transition(RefereeCommand.STOP, RefereeCommand.DIRECT_FREE_YELLOW) is ResetTier.BARRIER


def test_resuming_from_a_barrier_command_is_still_a_barrier():
    assert classify_transition(RefereeCommand.PREPARE_KICKOFF_YELLOW, RefereeCommand.NORMAL_START) is ResetTier.BARRIER
    assert classify_transition(RefereeCommand.BALL_PLACEMENT_BLUE, RefereeCommand.FORCE_START) is ResetTier.BARRIER


def test_resuming_from_a_pause_is_not_a_barrier():
    """STOP -> FORCE_START with no restart command in between must not reset mem."""
    assert classify_transition(RefereeCommand.STOP, RefereeCommand.FORCE_START) is ResetTier.NONE
    assert classify_transition(RefereeCommand.HALT, RefereeCommand.NORMAL_START) is ResetTier.NONE


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
