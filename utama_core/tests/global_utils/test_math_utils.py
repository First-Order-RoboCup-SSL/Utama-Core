import math

import numpy as np
import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import (
    angle_between_points,
    assert_valid_bounding_box,
    compute_bounding_zone_from_points,
    deg_to_rad,
    distance,
    normalise_heading,
    rad_to_deg,
    rotate_vector,
)

# --- Import the module under test ---


# -----------------------------------------------------------------
# rotate_vector
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "vx, vy, theta, expected",
    [
        (1, 0, 0, (1, 0)),  # no rotation
        (
            1,
            0,
            math.pi / 2,
            (0, -1),
        ),  # frame rotated +90° -> vector appears rotated -90° (CW)
        (
            0,
            1,
            -math.pi / 2,
            (-1, 0),
        ),  # frame rotated -90° -> vector appears +90° (CCW)
        (1, 1, math.pi, (-1, -1)),  # 180° rotation
    ],
)
def test_rotate_vector(vx, vy, theta, expected):
    vx_local, vy_local = rotate_vector(vx, vy, theta)
    # allow small numerical noise near zero
    np.testing.assert_allclose([vx_local, vy_local], expected, atol=1e-7, rtol=0)


# -----------------------------------------------------------------
# normalise_heading
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0, 0),
        (math.pi, math.pi),  # we’ll handle equivalence below
        (-math.pi, -math.pi),
        (3 * math.pi, math.pi),
        (-3 * math.pi, -math.pi),
        (2 * math.pi, 0),
    ],
)
def test_normalise_heading(angle, expected):
    result = normalise_heading(angle)

    # Handle π vs −π equivalence
    if math.isclose(abs(result), math.pi, abs_tol=1e-9) and math.isclose(abs(expected), math.pi, abs_tol=1e-9):
        assert True
    else:
        assert math.isclose(result, expected, abs_tol=1e-9)


# -----------------------------------------------------------------
# deg_to_rad / rad_to_deg
# -----------------------------------------------------------------


def test_deg_to_rad_and_rad_to_deg_roundtrip():
    deg = 180
    rad = deg_to_rad(deg)

    # Accept either π or -π since both represent 180° in [-π, π]
    assert math.isclose(abs(rad), math.pi, abs_tol=1e-9)

    deg_back = rad_to_deg(rad)
    assert math.isclose(deg_back % 360, 180.0, abs_tol=1e-9)


@pytest.mark.parametrize(
    "deg, expected_rad",
    [
        (0, 0),
        (90, math.pi / 2),
        (270, -math.pi / 2),
    ],
)
def test_deg_to_rad_values(deg, expected_rad):
    result = deg_to_rad(deg)
    assert math.isclose(result, expected_rad, abs_tol=1e-9)


@pytest.mark.parametrize(
    "rad, expected_deg",
    [
        (0, 0),
        (math.pi / 2, 90),
        (math.pi, 180),
        (-math.pi / 2, 270),  # normalized
    ],
)
def test_rad_to_deg_values(rad, expected_deg):
    result = rad_to_deg(rad)
    assert math.isclose(result, expected_deg, abs_tol=1e-9)


# -----------------------------------------------------------------
# distance
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "p1, p2, expected",
    [
        ((0, 0), (3, 4), 5),
        ((1, 1), (1, 1), 0),
        ((-1, -1), (2, 3), 5),
    ],
)
def test_distance(p1, p2, expected):
    result = distance(p1, p2)
    assert math.isclose(result, expected, abs_tol=1e-9)


# -----------------------------------------------------------------
# angle_between_points
# -----------------------------------------------------------------


def test_angle_between_points_right_angle():
    p0 = Vector2D(0, 0)
    p1 = Vector2D(1, 0)
    p2 = Vector2D(0, 1)
    result = angle_between_points(p0, p1, p2)
    assert math.isclose(result, math.pi / 2, abs_tol=1e-9)


def test_angle_between_points_straight_line():
    p0 = Vector2D(0, 0)
    p1 = Vector2D(1, 0)
    p2 = Vector2D(-1, 0)
    result = angle_between_points(p0, p1, p2)
    assert math.isclose(result, math.pi, abs_tol=1e-9)


# -----------------------------------------------------------------
# compute_bounding_zone_from_points
# -----------------------------------------------------------------


def test_compute_bounding_zone_from_points_basic():
    points = [(0, 0), (1, 2), (-2, 1)]
    bb = compute_bounding_zone_from_points(points)
    assert isinstance(bb, FieldBounds)
    assert bb.top_left == (-2, 2)
    assert bb.bottom_right == (1, 0)


def test_compute_bounding_zone_from_points_with_vector2d():
    points = [Vector2D(0, 0), Vector2D(1, 1), Vector2D(-1, -1)]
    bb = compute_bounding_zone_from_points(points)
    assert bb.top_left == (-1, 1)
    assert bb.bottom_right == (1, -1)


# -----------------------------------------------------------------
# assert_valid_bounding_box
# -----------------------------------------------------------------


def test_assert_valid_bounding_box_valid():
    bb = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    assert_valid_bounding_box(bb)  # Should not raise


@pytest.mark.parametrize(
    "top_left,bottom_right",
    [
        ((1, 1), (2, 2)),  # y0 < y1
        ((2, 2), (1, 1)),  # x0 > x1
        ((20, 0), (25, -1)),  # x out of field
        ((0, 20), (1, 25)),  # y out of field
    ],
)
def test_assert_valid_bounding_box_invalid(top_left, bottom_right):
    bb = FieldBounds(top_left, bottom_right)
    with pytest.raises(AssertionError):
        assert_valid_bounding_box(bb)
