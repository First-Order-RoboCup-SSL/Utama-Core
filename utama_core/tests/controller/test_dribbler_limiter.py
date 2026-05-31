"""Tests for the leaky-bucket dribbler thermal limiter in RealRobotController."""

from unittest.mock import MagicMock, patch

import pytest

from utama_core.team_controller.src.controllers.real.real_robot_controller import (
    DRIBBLER_MAX_ON_SECONDS,
    DRIBBLER_RESUME_SECONDS,
    RealRobotController,
)

_MONOTONIC = "utama_core.team_controller.src.controllers.real.real_robot_controller.time.monotonic"


@pytest.fixture
def controller():
    """RealRobotController with a mock serial port (no real hardware needed)."""
    mock_serial = MagicMock()
    mock_serial.in_waiting = 0
    return RealRobotController(is_team_yellow=True, n_friendly=3, serial_port=mock_serial)


def _tick(controller, robot_id: int, requested: bool, dt: float) -> bool:
    """Call _update_dribbler_bucket with a controlled time delta.

    Seeds _dribbler_last_tick so that time.monotonic() - last_tick == dt exactly.
    """
    now = controller._dribbler_last_tick.get(robot_id, 0.0) + dt
    controller._dribbler_last_tick[robot_id] = now - dt
    with patch(_MONOTONIC, return_value=now):
        return controller._update_dribbler_bucket(robot_id, requested)


# ---------------------------------------------------------------------------
# Basic bucket mechanics
# ---------------------------------------------------------------------------


def test_dribbler_allowed_when_bucket_empty(controller):
    assert _tick(controller, 0, requested=True, dt=1.0) is True


def test_bucket_fills_by_elapsed_seconds(controller):
    _tick(controller, 0, requested=True, dt=5.0)
    assert controller._dribbler_seconds[0] == pytest.approx(5.0)


def test_bucket_drains_by_elapsed_seconds(controller):
    controller._dribbler_seconds[0] = 20.0
    _tick(controller, 0, requested=False, dt=7.0)
    assert controller._dribbler_seconds[0] == pytest.approx(13.0)


def test_bucket_floors_at_zero(controller):
    controller._dribbler_seconds[0] = 2.0
    _tick(controller, 0, requested=False, dt=10.0)
    assert controller._dribbler_seconds[0] == pytest.approx(0.0)


def test_dribbler_off_returns_false(controller):
    assert _tick(controller, 0, requested=False, dt=1.0) is False


def test_bucket_does_not_overfill(controller):
    _tick(controller, 0, requested=True, dt=DRIBBLER_MAX_ON_SECONDS * 2)
    assert controller._dribbler_seconds[0] == pytest.approx(DRIBBLER_MAX_ON_SECONDS)


# ---------------------------------------------------------------------------
# Limit enforcement
# ---------------------------------------------------------------------------


def test_dribbler_forced_off_at_limit(controller):
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    assert _tick(controller, 0, requested=True, dt=1.0) is False


def test_dribbler_forced_off_emits_warning_once(controller):
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    # first tick at limit: warning
    with pytest.warns(UserWarning, match="thermal limit"):
        _tick(controller, 0, requested=True, dt=1.0)
    # subsequent ticks: no more warnings
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        _tick(controller, 0, requested=True, dt=1.0)  # must not raise


# ---------------------------------------------------------------------------
# Hysteresis: no 1s-on / 1s-off oscillation at the limit
# ---------------------------------------------------------------------------


def test_throttled_robot_stays_off_until_resume_threshold(controller):
    # fill to limit
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    _tick(controller, 0, requested=True, dt=0.0)  # trigger throttle flag

    # drain to just above resume threshold — still blocked
    controller._dribbler_seconds[0] = DRIBBLER_RESUME_SECONDS + 1.0
    assert _tick(controller, 0, requested=True, dt=0.0) is False


def test_throttled_robot_resumes_at_resume_threshold(controller):
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    _tick(controller, 0, requested=True, dt=0.0)  # trigger throttle

    # drain to exactly the resume threshold
    _tick(controller, 0, requested=False, dt=DRIBBLER_MAX_ON_SECONDS - DRIBBLER_RESUME_SECONDS)
    assert controller._dribbler_seconds[0] == pytest.approx(DRIBBLER_RESUME_SECONDS)
    # now requesting dribble should be allowed
    assert _tick(controller, 0, requested=True, dt=1.0) is True


def test_bucket_drains_while_throttled(controller):
    # robot should still drain even while strategy keeps requesting dribble
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    _tick(controller, 0, requested=True, dt=0.0)  # throttle
    _tick(controller, 0, requested=True, dt=5.0)
    assert controller._dribbler_seconds[0] == pytest.approx(DRIBBLER_MAX_ON_SECONDS - 5.0)


def test_no_oscillation_at_limit(controller):
    # fill to limit, then alternate on/off — robot must stay off until resume threshold
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    results = []
    for i in range(10):
        requested = i % 2 == 0  # alternate on/off
        results.append(_tick(controller, 0, requested=requested, dt=1.0))
    # all must be False until bucket drains below DRIBBLER_RESUME_SECONDS
    assert all(r is False for r in results)


# ---------------------------------------------------------------------------
# Recovery after full drain
# ---------------------------------------------------------------------------


def test_dribbler_fully_recovers_after_drain(controller):
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    _tick(controller, 0, requested=True, dt=0.0)  # throttle
    _tick(controller, 0, requested=False, dt=DRIBBLER_MAX_ON_SECONDS)
    assert controller._dribbler_seconds[0] == pytest.approx(0.0)
    assert _tick(controller, 0, requested=True, dt=1.0) is True


def test_limit_hit_at_correct_wall_time(controller):
    # 29s → allowed, fills to 29s
    assert _tick(controller, 0, requested=True, dt=29.0) is True
    assert controller._dribbler_seconds[0] == pytest.approx(29.0)
    # 1s more fills to 30s (cap) → still allowed (not yet at limit at entry)
    assert _tick(controller, 0, requested=True, dt=1.0) is True
    assert controller._dribbler_seconds[0] == pytest.approx(DRIBBLER_MAX_ON_SECONDS)
    # next tick: at limit → forced off
    assert _tick(controller, 0, requested=True, dt=1.0) is False


# ---------------------------------------------------------------------------
# Per-robot isolation
# ---------------------------------------------------------------------------


def test_buckets_are_independent_per_robot(controller):
    controller._dribbler_seconds[0] = DRIBBLER_MAX_ON_SECONDS
    assert _tick(controller, 0, requested=True, dt=0.0) is False  # throttle robot 0
    assert _tick(controller, 1, requested=True, dt=1.0) is True


def test_draining_one_robot_does_not_affect_another(controller):
    controller._dribbler_seconds[0] = 20.0
    controller._dribbler_seconds[1] = 10.0
    _tick(controller, 0, requested=False, dt=5.0)
    assert controller._dribbler_seconds[1] == pytest.approx(10.0)
