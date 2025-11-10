"""Obstacle helpers shared by the motion planners (NumPy version)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from utama_core.motion_planning.src.planning.geometry import AxisAlignedRectangle


@dataclass(frozen=True)
class ObstacleRegion:
    """Container describing a temporary obstacle.

    Attributes:
        polygon: NumPy (N, 2) array of (x, y) coordinates describing the obstacle boundary.
        rect: A cheap axis-aligned bounding box approximation used in the fast
            planner for clearance checks.
    """

    polygon: np.ndarray  # shape (N, 2)
    rect: AxisAlignedRectangle

    @classmethod
    def from_polygon(cls, polygon: np.ndarray) -> "ObstacleRegion":
        """Create ObstacleRegion from a polygon array."""
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("Polygon must be a 2D NumPy array of shape (N, 2).")

        min_x, min_y = polygon.min(axis=0)
        max_x, max_y = polygon.max(axis=0)
        return cls(polygon=polygon, rect=AxisAlignedRectangle(min_x, max_x, min_y, max_y))


def to_polygons(obstacles: Iterable[ObstacleRegion]) -> List[np.ndarray]:
    """Return the list of polygon arrays for the given obstacles."""
    return [obs.polygon for obs in obstacles]


def to_rectangles(obstacles: Iterable[ObstacleRegion]) -> List[AxisAlignedRectangle]:
    """Return the list of bounding rectangles for the given obstacles."""
    return [obs.rect for obs in obstacles]
