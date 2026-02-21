from typing import Tuple

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import Field, FieldBounds


def rotate_vector(vx_global: float, vy_global: float, theta: float) -> Tuple[float, float]:
    """Rotates a 2D vector from global coordinates to local coordinates based on a given angle.

    Args:
        vx_global (float): The x-component of the vector in the global coordinate system.
        vy_global (float): The y-component of the vector in the global coordinate system.
        theta (float): The angle in radians to rotate the vector, typically representing the orientation of a local frame.

    Returns:
        Tuple[float, float]: A tuple containing the x and y components of the vector in the local coordinate system.

    The function uses a 2D rotation matrix to transform the vector from global to local coordinates, where:
        - `vx_local` = vx_global * cos(theta) + vy_global * sin(theta)
        - `vy_local` = -vx_global * sin(theta) + vy_global * cos(theta)
    """
    vx_local = vx_global * np.cos(theta) + vy_global * np.sin(theta)
    vy_local = -vx_global * np.sin(theta) + vy_global * np.cos(theta)
    return vx_local, vy_local


def normalise_heading(angle):
    """Normalize an angle to the range [-π, π] radians, where 0 faces along positive x-axis.

    Parameters
    ----------
    angle : float
        The angle in radians to be normalized. The input angle can be any real number.

    Returns
    -------
    float
        The normalized angle in the range [-π, π] radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalise_heading_deg(angle):
    """Normalize an angle to the range [-180, 180] degrees, where 0 faces along positive x-axis.

    Parameters
    ----------
    angle : float
        The angle in degrees to be normalized. The input angle can be any real number.

    Returns
    -------
    float
        The normalized angle in the range [-180, 180] degrees.
    """
    half_rev = 180

    return (angle + half_rev) % (2 * half_rev) - half_rev


def deg_to_rad(degrees: float):
    """Convert degrees to radians, then normalise to range [-π, π]"""
    radians = np.deg2rad(degrees)
    return normalise_heading(radians)


def rad_to_deg(radians: float):
    """Convert radians to degrees, then normalise to range [0, 360]"""
    degrees = np.rad2deg(radians)
    return degrees % 360


def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points in 2D space using the Pythagorean theorem."""
    return np.hypot(point2[0] - point1[0], point2[1] - point1[1])


def angle_between_points(main_point: Vector2D, point1: Vector2D, point2: Vector2D):
    """Computes the angle (in radians) between two lines originating from main_point and passing through point1 and
    point2.

    Parameters:
    main_point (tuple): The common point (x, y).
    point1 (tuple): First point (x, y).
    point2 (tuple): Second point (x, y).

    Returns:
    float: Angle in degrees between the two lines.
    """
    v1 = point1 - main_point
    v2 = point2 - main_point
    return v1.angle_between(v2)


def compute_bounding_zone_from_points(
    points: list[Tuple[float, float] | Vector2D | np.ndarray],
) -> FieldBounds:
    """Compute the minimum bounding zone that contains all given points.

    Args:
        points (list of tuple): A list of (x, y) tuples representing the points.

    Returns:
        FieldBounds: The minimum bounding zone containing all points.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return FieldBounds(top_left=(min_x, max_y), bottom_right=(max_x, min_y))


def assert_valid_bounding_box(bb: FieldBounds):
    """Asserts that a FieldBounds object is valid, raising an AssertionError if not."""
    fx, fy = Field._FULL_FIELD_HALF_LENGTH, Field._FULL_FIELD_HALF_WIDTH

    x0, y0 = bb.top_left
    x1, y1 = bb.bottom_right
    assert x0 <= x1, f"top-left x {x0} must be <= bottom-right x {x1}"
    assert y0 >= y1, f"top-left y {y0} must be >= bottom-right y {y1}"
    # Also ensure within full field
    assert -fx <= x0 <= fx and -fx <= x1 <= fx, f"x coordinates out of full field bounds ±{fx}"
    assert -fy <= y0 <= fy and -fy <= y1 <= fy, f"y coordinates out of full field bounds ±{fy}"
