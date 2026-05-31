"""Tests for the leaky-bucket dribbler thermal limiter in RealRobotController."""

from unittest.mock import MagicMock

import pytest

from utama_core.team_controller.src.controllers.real.real_robot_controller import (
    DRIBBLER_MAX_ON_STEPS,
    RealRobotController,
)


@pytest.fixture
def controller():
    """RealRobotController with a mock serial port (no real hardware needed)."""
    mock_serial = MagicMock()
    mock_serial.in_waiting = 0
    ctrl = RealRobotController(
        is_team_yellow=True,
        n_friendly=3,
        serial_port=mock_serial,
    )
    return ctrl


# ---------------------------------------------------------------------------
# Basic bucket mechanics
# ---------------------------------------------------------------------------


def test_dribbler_allowed_when_bucket_empty(controller):
    assert controller._update_dribbler_bucket(0, requested=True) is True


def test_bucket_increments_while_dribbling(controller):
    for _ in range(10):
        controller._update_dribbler_bucket(0, requested=True)
    assert controller._dribbler_steps[0] == 10


def test_bucket_decrements_while_not_dribbling(controller):
    # fill to 20 steps
    for _ in range(20):
        controller._update_dribbler_bucket(0, requested=True)
    # drain for 8 steps
    for _ in range(8):
        controller._update_dribbler_bucket(0, requested=False)
    assert controller._dribbler_steps[0] == 12


def test_bucket_floors_at_zero(controller):
    for _ in range(5):
        controller._update_dribbler_bucket(0, requested=False)
    assert controller._dribbler_steps.get(0, 0) == 0


def test_dribbler_off_returns_false(controller):
    assert controller._update_dribbler_bucket(0, requested=False) is False


# ---------------------------------------------------------------------------
# Limit enforcement
# ---------------------------------------------------------------------------


def test_dribbler_forced_off_at_limit(controller):
    # fill bucket to exactly the limit
    controller._dribbler_steps[0] = DRIBBLER_MAX_ON_STEPS
    assert controller._update_dribbler_bucket(0, requested=True) is False


def test_dribbler_forced_off_emits_warning(controller):
    controller._dribbler_steps[0] = DRIBBLER_MAX_ON_STEPS
    with pytest.warns(UserWarning, match="thermal limit"):
        controller._update_dribbler_bucket(0, requested=True)


def test_dribbler_recovers_after_draining(controller):
    # hit the limit
    controller._dribbler_steps[0] = DRIBBLER_MAX_ON_STEPS
    assert controller._update_dribbler_bucket(0, requested=True) is False

    # drain fully
    for _ in range(DRIBBLER_MAX_ON_STEPS):
        controller._update_dribbler_bucket(0, requested=False)

    assert controller._dribbler_steps[0] == 0
    assert controller._update_dribbler_bucket(0, requested=True) is True


def test_bucket_does_not_exceed_limit_while_draining(controller):
    # At limit, requesting off should drain, not stay stuck
    controller._dribbler_steps[0] = DRIBBLER_MAX_ON_STEPS
    controller._update_dribbler_bucket(0, requested=False)
    assert controller._dribbler_steps[0] == DRIBBLER_MAX_ON_STEPS - 1


# ---------------------------------------------------------------------------
# Leaky-bucket proportionality: intermittent use
# ---------------------------------------------------------------------------


def test_intermittent_use_drains_proportionally(controller):
    # 20s on, 10s off → bucket should be at 600 steps (10s worth)
    on_steps = 20 * 60
    off_steps = 10 * 60
    for _ in range(on_steps):
        controller._update_dribbler_bucket(0, requested=True)
    for _ in range(off_steps):
        controller._update_dribbler_bucket(0, requested=False)
    assert controller._dribbler_steps[0] == on_steps - off_steps


def test_full_cycle_hits_limit_at_expected_step(controller):
    # With an empty bucket, dribbler should be forced off at exactly step DRIBBLER_MAX_ON_STEPS
    forced_off_at = None
    for i in range(DRIBBLER_MAX_ON_STEPS + 5):
        result = controller._update_dribbler_bucket(0, requested=True)
        if not result and forced_off_at is None:
            forced_off_at = i
    assert forced_off_at == DRIBBLER_MAX_ON_STEPS


# ---------------------------------------------------------------------------
# Per-robot isolation
# ---------------------------------------------------------------------------


def test_buckets_are_independent_per_robot(controller):
    # Fill robot 0's bucket to limit; robot 1 should be unaffected
    controller._dribbler_steps[0] = DRIBBLER_MAX_ON_STEPS
    assert controller._update_dribbler_bucket(0, requested=True) is False
    assert controller._update_dribbler_bucket(1, requested=True) is True


def test_draining_one_robot_does_not_affect_another(controller):
    controller._dribbler_steps[0] = 100
    controller._dribbler_steps[1] = 50
    controller._update_dribbler_bucket(0, requested=False)
    assert controller._dribbler_steps[1] == 50
