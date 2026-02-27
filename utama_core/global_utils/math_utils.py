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


def distance_between_line_segments(
    seg1_start: np.ndarray,
    seg1_end: np.ndarray,
    seg2_start: np.ndarray,
    seg2_end: np.ndarray,
) -> float:
    """Calculate the minimum distance between two line segments in 2D space.

    Args:
        seg1_start (tuple): A tuple representing the start of the first line segment (x1, y1).
        seg1_end (tuple): A tuple representing the end of the first line segment (x2, y2).
        seg2_start (tuple): A tuple representing the start of the second line segment (x3, y3).
        seg2_end (tuple): A tuple representing the end of the second line segment (x4, y4).
    Returns:
        float: The minimum distance between the two line segments.
    """
    if segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
        return 0.0

    return min(
        distance_point_to_segment(seg1_start, seg2_start, seg2_end),
        distance_point_to_segment(seg1_end, seg2_start, seg2_end),
        distance_point_to_segment(seg2_start, seg1_start, seg1_end),
        distance_point_to_segment(seg2_end, seg1_start, seg1_end),
    )


def distance_point_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Calculate the minimum distance from a point to a line segment in 2D space.

    Args:
        point (tuple): A tuple representing the point (px, py).
        seg_start (tuple): A tuple representing the start of the segment (x1, y1).
        seg_end (tuple): A tuple representing the end of the segment (x2, y2).
    Returns:
        float: The minimum distance from the point to the line segment.
    """
    x_1, y_1 = point
    (x_2, y_2), (x_3, y_3) = seg_start, seg_end

    dx = x_3 - x_2
    dy = y_3 - y_2

    if dx == dy == 0:  # e.g. for static object the segment will be a point
        return np.hypot(x_1 - x_2, y_1 - y_2)

    # Now assume the segment is a line, calculate minimum distance to the line
    # t describes the point on this line closest to point, in the form
    # p_closest = (x_2 + t*dx, y_2 + t*dy),
    # where (dx, dy) is the line direction vector
    t = ((x_1 - x_2) * dx + (y_1 - y_2) * dy) / (dx * dx + dy * dy)

    if t < 0:
        # Closest point is the start of the segment
        closest_x, closest_y = x_2, y_2
    elif t > 1:
        # Closest point is the end of the segment
        closest_x, closest_y = x_3, y_3
    else:
        # Closest point is within the segment
        closest_x = x_2 + t * dx
        closest_y = y_2 + t * dy

    return np.hypot(x_1 - closest_x, y_1 - closest_y)


def segments_intersect(
    seg1_start: np.ndarray,
    seg1_end: np.ndarray,
    seg2_start: np.ndarray,
    seg2_end: np.ndarray,
):
    """Check if two line segments intersect.

    Args:
        seg1_start (tuple): ((x1, y1), (x2, y2))
        seg1_end (tuple): ((x3, y3), (x4, y4))
        seg2_start (tuple): ((x5, y5), (x6, y6))
        seg2_end (tuple): ((x7, y7), (x8, y8))
    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    (p1, q1) = seg1_start, seg1_end
    (p2, q2) = seg2_start, seg2_end

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases (collinear cases where they touch/overlap)
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def orientation(p_1: np.ndarray, p_2: np.ndarray, p_3: np.ndarray) -> int:
    """Calculate the orientation of 3 points (e.g. on a line or in a triangle).

    Args:
        p_1 (np.ndarray): First point as (x, y).
        p_2 (np.ndarray): Second point as (x, y).
        p_3 (np.ndarray): Third point as (x, y).

    Returns:
        int: 0 if collinear, 1 if clockwise, 2 if counterclockwise.
    """
    val = (p_2[1] - p_1[1]) * (p_3[0] - p_2[0]) - (p_2[0] - p_1[0]) * (p_3[1] - p_2[1])
    if val == 0:
        return 0  # collinear

    return 1 if val > 0 else 2  # 1: clockwise, 2: counterclockwise


def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """Check if point q lies on line segment 'pr'.

    Args:
        p (np.ndarray): Start point of segment as (x, y).
        q (np.ndarray): Point to check as (x, y).
        r (np.ndarray): End point of segment as (x, y).

    Returns:
        bool: True if q lies on segment pr, False otherwise.
    """
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True

    return False


def find_intersection(line1, line2):
    """
    Find the intersection point of two line segments.

    Args:
        line1: tuple of two np.arrays (start, end) -> (A, B)
        line2: tuple of two np.arrays (start, end) -> (C, D)

    Returns:
        np.array of intersection point (x, y), or None if no intersection.
    """
    A, B = line1
    C, D = line2

    denom = (B[0] - A[0]) * (D[1] - C[1]) - (B[1] - A[1]) * (D[0] - C[0])

    if denom == 0:
        return None  # Parallel lines

    t = ((C[0] - A[0]) * (D[1] - C[1]) - (C[1] - A[1]) * (D[0] - C[0])) / denom
    u = ((C[0] - A[0]) * (B[1] - A[1]) - (C[1] - A[1]) * (B[0] - A[0])) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return A + t * (B - A)

    return None  # Segments don't intersect
