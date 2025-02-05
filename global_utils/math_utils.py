import numpy as np
from typing import Tuple


def rotate_vector(
    vx_global: float, vy_global: float, theta: float
) -> Tuple[float, float]:
    """
    Rotates a 2D vector from global coordinates to local coordinates based on a given angle.

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
    """
    Normalize an angle to the range [-π, π] radians, where 0 faces along positive x-axis.

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


def deg_to_rad(degrees: float):
    """
    Convert degrees to radians, then normalise to range [-π, π]
    """
    radians = np.deg2rad(degrees)
    return normalise_heading(radians)


def rad_to_deg(radians: float):
    """
    Convert radians to degrees, then normalise to range [0, 360]
    """
    degrees = np.rad2deg(radians)
    return degrees % 360


def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    Parameters:
        point1 (Tuple[float, float]): Coordinates of the first point (x1, y1).
        point2 (Tuple[float, float]): Coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    Parameters:
        point1 (Tuple[float, float]): Coordinates of the first point (x1, y1).
        point2 (Tuple[float, float]): Coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2D points.

    Parameters:
        point1 (Tuple[float, float]): Coordinates of the first point (x1, y1).
        point2 (Tuple[float, float]): Coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
