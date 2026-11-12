"""The last metre of a move: speed floor under `TwoDPID`, leaving cruising and settling to kp."""

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.motion_planning.pid.configs import TranslationPIDConfigs
from utama_core.motion_planning.pid.pid import TwoDPID

_ORIGIN = Vector2D(0.0, 0.0)
_KP = 1.8


def _speed(distance: float, fraction: float = 0.6, has_final_target: bool = True) -> float:
    pid = TwoDPID(
        TranslationPIDConfigs(max_velocity=2.0, kp=_KP, max_acceleration=4.0, approach_brake_fraction=fraction)
    )
    target = Vector2D(distance, 0.0)
    if has_final_target:
        pid.set_final_target(0, target)
    return pid._calculate(target, _ORIGIN, 0).mag()


def test_the_last_metre_holds_a_share_of_the_speed_it_can_stop_from():
    assert _speed(0.25) == pytest.approx(0.6 * (2 * 4.0 * 0.25) ** 0.5)


def test_the_floor_starts_just_inside_a_metre():
    assert _speed(0.99) > _KP * 0.99 - 1e-9
    assert _speed(1.01) == pytest.approx(_KP * 1.01, rel=0.02)


def test_the_floor_never_exceeds_cruising_speed():
    assert _speed(0.95, fraction=1.0) <= _KP + 1e-9


def test_kp_settles_the_robot_inside_the_last_ten_centimetres():
    assert _speed(0.09) == pytest.approx(_KP * 0.09, rel=0.02)
    assert _speed(0.11) == pytest.approx(0.6 * (2 * 4.0 * 0.11) ** 0.5)


def test_a_fraction_of_zero_leaves_the_speed_to_kp():
    assert _speed(0.25, fraction=0.0) == pytest.approx(_KP * 0.25, rel=0.02)


def test_a_caller_without_a_final_target_is_left_to_kp():
    assert _speed(0.25, has_final_target=False) == pytest.approx(_KP * 0.25, rel=0.02)


def test_a_waypoint_nearer_than_the_destination_is_left_to_kp():
    def speed(destination_x: float) -> float:
        pid = TwoDPID(
            TranslationPIDConfigs(max_velocity=2.0, kp=_KP, max_acceleration=4.0, approach_brake_fraction=0.6)
        )
        pid.set_final_target(0, Vector2D(destination_x, 0.0))
        return pid._calculate(Vector2D(0.25, 0.0), _ORIGIN, 0).mag()

    assert speed(0.26) > _KP * 0.25 + 0.1  # 1 cm past the waypoint: still a straight run
    assert speed(0.28) == pytest.approx(_KP * 0.25, rel=0.02)  # 3 cm: a corner
