"""Obstacle helpers shared by the motion planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from shapely import Polygon

from utama_core.motion_planning.src.planning.geometry import AxisAlignedRectangle


@dataclass(frozen=True)
class ObstacleRegion:
    """Container describing a temporary obstacle.

    Attributes:
        polygon: Original Shapely representation (used by the slower planners
            and exit strategies that still rely on precise polygon operations).
        rect: A cheap axis-aligned bounding box approximation used in the fast
            planner for clearance checks.
    """

    polygon: Polygon
    rect: AxisAlignedRectangle

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> "ObstacleRegion":
        min_x, min_y, max_x, max_y = polygon.bounds
        return cls(polygon=polygon, rect=AxisAlignedRectangle(min_x, max_x, min_y, max_y))


def to_polygons(obstacles: Iterable[ObstacleRegion]) -> List[Polygon]:
    return [obs.polygon for obs in obstacles]


def to_rectangles(obstacles: Iterable[ObstacleRegion]) -> List[AxisAlignedRectangle]:
    return [obs.rect for obs in obstacles]
