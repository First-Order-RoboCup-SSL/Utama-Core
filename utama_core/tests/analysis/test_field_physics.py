"""Unit tests for tools/field_physics.py: the register filter, step delays, ball rolls, agreement.

No real logs and no rsim: the ball roll is checked on a synthetic ball with a known deceleration.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from tools import field_physics


def _group(median: float, n: int) -> dict:
    return {"median": median, "lo": median, "hi": median, "n": n}


# allowed


def test_allowed_listed_measurement_true_unlisted_false():
    register = {"sessions": {"20261028": {"use": ["loop_delay", "ball_roll"]}}}
    assert field_physics.allowed(register, "20261028", "loop_delay") is True
    assert field_physics.allowed(register, "20261028", "ball_noise") is False


def test_allowed_unregistered_day_false():
    register = {"sessions": {"20261028": {"use": ["loop_delay"]}}}
    assert field_physics.allowed(register, "20261030", "loop_delay") is False


def test_allowed_on_real_register():
    register = field_physics.load_register()
    for measurement in field_physics.MEASUREMENTS:
        assert field_physics.allowed(register, "20261021", measurement) is False
    assert field_physics.allowed(register, "20261029", "loop_delay") is False
    assert field_physics.allowed(register, "20261029", "ball_noise") is True


# step_delays_ms


def test_step_delays_commanded_boundary():
    steps = [
        {"axis": "forward", "commanded": 0.3, "delay_s": 0.05},
        {"axis": "forward", "commanded": 0.29, "delay_s": 0.05},
    ]
    assert field_physics.step_delays_ms(steps) == pytest.approx([50.0])


def test_step_delays_excludes_turn_zero_delay_and_missing_delay():
    steps = [
        {"axis": "turn", "commanded": 1.0, "delay_s": 0.05},
        {"axis": "left", "commanded": -0.5, "delay_s": 0.06},
        {"axis": "forward", "commanded": 0.5, "delay_s": 0.0},
        {"axis": "forward", "commanded": 0.5},
    ]
    assert field_physics.step_delays_ms(steps) == pytest.approx([60.0])


# ball_rolls


def _synthetic_log(robots: dict | None = None, duration: float = 4.0):
    """A ball rolling from x=0 at 2 m/s with deceleration 0.25 m/s^2, seen at 60 Hz by one camera."""
    t = np.arange(0.0, duration, 1 / 60)
    x = 2.0 * t - 0.5 * 0.25 * t**2
    balls = np.column_stack([t, np.zeros_like(t), x, np.zeros_like(t)])
    return types.SimpleNamespace(balls=balls, robots=robots or {})


BOUNDS = ((-4.5, 3.0), (4.5, -3.0))
KICK = [{"t": 0.0, "ball_speed": 2.0, "chip": False}]


def test_ball_rolls_recovers_known_decel():
    rolls = field_physics.ball_rolls(_synthetic_log(), KICK, BOUNDS)
    assert len(rolls) == 1
    assert rolls[0]["decel"] == pytest.approx(0.25, abs=0.01)
    assert rolls[0]["rms_mm"] < 1.0


def test_ball_rolls_truncated_by_robot_in_path_does_not_crash():
    # A robot 2.5 m along the ball's path stops the fit early: a shorter fit, still the right decel.
    robot = np.column_stack(
        [np.arange(0.0, 4.0, 1 / 60), np.zeros(240), np.full(240, 2.5), np.zeros(240), np.zeros(240)]
    )
    rolls = field_physics.ball_rolls(_synthetic_log(robots={("yellow", 3): robot}), KICK, BOUNDS)
    assert len(rolls) == 1
    assert rolls[0]["decel"] == pytest.approx(0.25, abs=0.01)
    assert rolls[0]["secs"] < 1.5


def test_ball_rolls_robot_at_half_metre_leaves_no_fit():
    # A robot 0.5 m along the path is already inside the 0.25 m exclusion when the fit starts (0.3 s):
    # no fit, and no crash.
    robot = np.column_stack(
        [np.arange(0.0, 4.0, 1 / 60), np.zeros(240), np.full(240, 0.5), np.zeros(240), np.zeros(240)]
    )
    assert field_physics.ball_rolls(_synthetic_log(robots={("yellow", 3): robot}), KICK, BOUNDS) == []


# agreement


def test_agreement_close_medians_agree():
    groups = {("20261028", 0): _group(0.20, 10), ("20261027", None): _group(0.25, 10)}
    res = field_physics.agreement(groups, tolerance=0.25)
    # highest - lowest = 0.05 must be <= 0.25 * median(0.20, 0.25) = 0.05625
    assert 0.25 - 0.20 <= 0.25 * np.median([0.20, 0.25])
    assert res["agree"] is True
    assert res["medians"] == [0.20, 0.25]
    assert res["groups_used"] == 2


def test_agreement_exact_boundary_agrees():
    # 9 - 7 = 2.0 equals 0.25 * median(7, 9) = 2.0: exactly equal counts as agreeing.
    groups = {("a", 0): _group(7.0, 10), ("b", 0): _group(9.0, 10)}
    assert field_physics.agreement(groups, tolerance=0.25)["agree"] is True


def test_agreement_just_past_boundary_disagrees():
    groups = {("a", 0): _group(7.0, 10), ("b", 0): _group(9.01, 10)}
    assert field_physics.agreement(groups, tolerance=0.25)["agree"] is False


def test_agreement_clearly_larger_spread_disagrees():
    groups = {("20261028", 0): _group(0.20, 10), ("20261027", None): _group(0.40, 10)}
    res = field_physics.agreement(groups, tolerance=0.25)
    assert res["agree"] is False
    assert res["groups_used"] == 2


def test_agreement_ignores_groups_below_min_n():
    groups = {
        ("20261028", 0): _group(0.20, 10),
        ("20261027", None): _group(0.25, 10),
        ("20261028", 1): _group(0.90, 4),
    }
    res = field_physics.agreement(groups)
    assert res["groups_used"] == 2
    assert res["medians"] == [0.20, 0.25]
    assert res["agree"] is True


def test_agreement_one_usable_group_is_none():
    groups = {("20261028", 0): _group(0.20, 10), ("20261028", 1): _group(0.90, 4)}
    res = field_physics.agreement(groups)
    assert res["agree"] is None
    assert res["groups_used"] == 1
