import math
from timeit import timeit

import numpy as np
import pytest

from utama_core.entities.data.vector import Vector2D, Vector3D

N_ITERS = 10000


def test_vector2d_operations():
    v1 = Vector2D(3, 4)
    v2 = Vector2D(1, 2)

    # Test addition
    v3 = v1 + v2
    assert v3.x == 4 and v3.y == 6

    # Test subtraction
    v4 = v1 - v2
    assert v4.x == 2 and v4.y == 2

    # Test scalar multiplication
    v5 = v1 * 2
    assert v5.x == 6 and v5.y == 8

    # Test dot product
    dot_product = np.dot(v1, v2)
    assert dot_product == 11

    # Test distance
    distance = v1.distance_to(v2)
    expected_distance = math.sqrt((3 - 1) ** 2 + (4 - 2) ** 2)
    assert math.isclose(distance, expected_distance)

    # Test angle
    angle = v1.angle_to(v2)
    expected_angle = math.atan2(2 - 4, 1 - 3)
    assert math.isclose(angle, expected_angle)

    mag = v1.mag()
    expected_mag = math.hypot(3, 4)
    assert math.isclose(mag, expected_mag)

    norm = v1.norm()
    expected_norm = Vector2D(3 / 5, 4 / 5)
    assert math.isclose(norm.x, expected_norm.x) and math.isclose(norm.y, expected_norm.y)


def test_vector3d_operations():
    v1 = Vector3D(3, 4, 5)
    v2 = Vector3D(1, 2, 3)

    # Test addition
    v3 = v1 + v2
    assert v3.x == 4 and v3.y == 6 and v3.z == 8

    # Test subtraction
    v4 = v1 - v2
    assert v4.x == 2 and v4.y == 2 and v4.z == 2

    # Test scalar multiplication
    v5 = v1 * 2
    assert v5.x == 6 and v5.y == 8 and v5.z == 10

    # Test dot product
    dot_product = np.dot(v1, v2)
    assert dot_product == 26

    # Test distance
    distance = math.sqrt((3 - 1) ** 2 + (4 - 2) ** 2 + (5 - 3) ** 2)
    calculated_distance = np.linalg.norm(v1 - v2)
    assert math.isclose(calculated_distance, distance)


def test_vector2d_and_vector3d_conversion():
    v3d = Vector3D(3, 4, 5)
    v2d = v3d.to_2d()
    assert v2d.x == 3 and v2d.y == 4


def test_vector2d_with_vector3d_operations():
    v2d = Vector2D(3, 4)
    v3d = Vector3D(1, 2, 3)

    # Test addition
    v4 = v2d + v3d.to_2d()
    assert v4.x == 4 and v4.y == 6

    # Test subtraction
    v5 = v2d - v3d.to_2d()
    assert v5.x == 2 and v5.y == 2

    # Test dot product
    dot_product = np.dot(v2d, v3d.to_2d())
    assert dot_product == 11

    xy_dist = v2d.distance_to(v3d)  # Should not raise any error
    assert xy_dist == math.sqrt((3 - 1) ** 2 + (4 - 2) ** 2)

    xy_angle = v2d.angle_to(v3d)  # Should not raise any error
    assert xy_angle == math.atan2(2 - 4, 1 - 3)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.shape != v2.shape:
        raise ValueError("Cannot calculate angle between vectors of different dimensions")
    dot_product = np.dot(v1, v2)
    mag_self = np.linalg.norm(v1)
    mag_other = np.linalg.norm(v2)
    norm_prod = mag_self * mag_other
    if norm_prod == 0:
        return 0.0
    cos_theta = dot_product / norm_prod
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def test_performance_against_numpy():
    v2d_1 = Vector2D(3, 4)
    v2d_2 = Vector2D(1, 2)

    np_arr_1 = np.array([3.0, 4.0])
    np_arr_2 = np.array([1.0, 2.0])

    # Test mag performance
    vector_mag_time = timeit(lambda: v2d_1.mag(), number=N_ITERS)
    numpy_mag_time = timeit(lambda: np.linalg.norm(np_arr_1), number=N_ITERS)
    assert vector_mag_time < numpy_mag_time

    # test angle between performance
    vector_angle_time = timeit(lambda: v2d_1.angle_between(v2d_2), number=N_ITERS)
    numpy_angle_time = timeit(lambda: angle_between(np_arr_1, np_arr_2), number=N_ITERS)
    assert vector_angle_time < numpy_angle_time

    # Test dot product performance
    vector_dot_time = timeit(lambda: v2d_1.distance_to(v2d_2), number=N_ITERS)
    numpy_dot_time = timeit(lambda: np.linalg.norm(np_arr_2 - np_arr_1), number=N_ITERS)
    assert vector_dot_time < numpy_dot_time
