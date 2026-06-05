import math

import pytest

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.skills.src.go_to_ball import (
    _APPROACH_OVERSHOOT_M,
    _DRIBBLE_OVERSHOOT_M,
    _target_past_ball,
)


def test_target_past_ball_overshoots_along_approach_line():
    ball = Vector2D(1.0, 0.0)

    target = _target_past_ball(ball, 0.0, 0.05)

    assert target.x == pytest.approx(1.05)
    assert target.y == pytest.approx(0.0)


def test_target_past_ball_overshoots_diagonal_approach():
    ball = Vector2D(1.0, 1.0)

    target = _target_past_ball(ball, math.atan2(4.0, 3.0), 0.05)

    assert target.x == pytest.approx(1.03)
    assert target.y == pytest.approx(1.04)


def test_dribbler_off_overshoot_is_smaller_than_dribbler_on():
    assert _APPROACH_OVERSHOOT_M == pytest.approx(ROBOT_RADIUS * 0.5)
    assert _DRIBBLE_OVERSHOOT_M > 0.0
    assert _APPROACH_OVERSHOOT_M < _DRIBBLE_OVERSHOOT_M
