"""Lightweight geometry helpers used by the motion-planning stack.

The Dynamic Window planner previously relied on Shapely primitives inside the
per-frame evaluation loop. To reduce per-step overhead we replace those calls
with simple numeric utilities implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

EPSILON = 1e-9


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray, eps: float = 1e-9) -> float:
    """Optimized minimal Euclidean distance between a point and a line segment (2D)."""
    segment = end - start
    denom = np.dot(segment, segment)

    if denom < eps:
        # Degenerate segment (start == end)
        diff = point - start
        return np.sqrt(np.dot(diff, diff))

    t = np.dot(point - start, segment) / denom
    # Clamp t without calling another function
    t = np.clip(t, 0.0, 1.0)
    projection = start + float(t) * segment
    diff = point - projection
    return np.sqrt(np.dot(diff, diff))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return (
        min(a[0], c[0]) - EPSILON <= b[0] <= max(a[0], c[0]) + EPSILON
        and min(a[1], c[1]) - EPSILON <= b[1] <= max(a[1], c[1]) + EPSILON
    )


def segments_intersect(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> bool:
    """Return True if the two closed segments intersect."""

    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    if abs(o1) <= EPSILON and _on_segment(p1, p2, q1):
        return True
    if abs(o2) <= EPSILON and _on_segment(p1, q2, q1):
        return True
    if abs(o3) <= EPSILON and _on_segment(p2, p1, q2):
        return True
    if abs(o4) <= EPSILON and _on_segment(p2, q1, q2):
        return True

    return False


def segment_to_segment_distance(
    a_start: np.ndarray,
    a_end: np.ndarray,
    b_start: np.ndarray,
    b_end: np.ndarray,
) -> float:
    if segments_intersect(a_start, a_end, b_start, b_end):
        return 0.0

    return min(
        point_segment_distance(a_start, b_start, b_end),
        point_segment_distance(a_end, b_start, b_end),
        point_segment_distance(b_start, a_start, a_end),
        point_segment_distance(b_end, a_start, a_end),
    )


@dataclass(frozen=True)
class AxisAlignedRectangle:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, point: Tuple[float, float]) -> bool:
        px, py = point
        return self.min_x <= px <= self.max_x and self.min_y <= py <= self.max_y

    def contains_array(self, point: np.ndarray) -> bool:
        return self.contains((float(point[0]), float(point[1])))

    def distance_to_boundary(self, point: Tuple[float, float]) -> float:
        px, py = point
        dx = max(self.min_x - px, 0.0, px - self.max_x)
        dy = max(self.min_y - py, 0.0, py - self.max_y)
        return math.hypot(dx, dy)

    def distance_to_boundary_array(self, point: np.ndarray) -> float:
        return self.distance_to_boundary((float(point[0]), float(point[1])))

    def corners(self) -> Iterable[np.ndarray]:
        return (
            np.array([self.min_x, self.min_y], dtype=float),
            np.array([self.max_x, self.min_y], dtype=float),
            np.array([self.max_x, self.max_y], dtype=float),
            np.array([self.min_x, self.max_y], dtype=float),
        )

    def _nearest_boundary_point_and_normal(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        px, py = float(point[0]), float(point[1])
        if self.contains((px, py)):
            distances = [
                (px - self.min_x, np.array([-1.0, 0.0]), np.array([self.min_x, py])),
                (self.max_x - px, np.array([1.0, 0.0]), np.array([self.max_x, py])),
                (py - self.min_y, np.array([0.0, -1.0]), np.array([px, self.min_y])),
                (self.max_y - py, np.array([0.0, 1.0]), np.array([px, self.max_y])),
            ]
            _, normal, boundary_point = min(distances, key=lambda entry: entry[0])
            return boundary_point.astype(float), normal

        clamped = np.array(
            [
                _clamp(px, self.min_x, self.max_x),
                _clamp(py, self.min_y, self.max_y),
            ]
        )
        diff = np.array([px, py]) - clamped
        norm = np.linalg.norm(diff)
        if norm < EPSILON:
            # Point lies exactly on an edge extension; choose normal based on proximity.
            if abs(px - self.min_x) < EPSILON:
                normal = np.array([-1.0, 0.0])
            elif abs(px - self.max_x) < EPSILON:
                normal = np.array([1.0, 0.0])
            elif abs(py - self.min_y) < EPSILON:
                normal = np.array([0.0, -1.0])
            else:
                normal = np.array([0.0, 1.0])
        else:
            normal = diff / norm
        return clamped, normal

    def exit_point_with_buffer(self, point: Tuple[float, float], buffer: float) -> Tuple[float, float]:
        point_arr = np.array(point, dtype=float)
        boundary_point, normal = self._nearest_boundary_point_and_normal(point_arr)
        target = boundary_point + buffer * normal
        return float(target[0]), float(target[1])

    def distance_to_segment(self, start: np.ndarray, end: np.ndarray) -> float:
        if self.contains_array(start) or self.contains_array(end):
            return 0.0

        distances = [
            self.distance_to_boundary_array(start),
            self.distance_to_boundary_array(end),
        ]

        for edge_start, edge_end in self._edges():
            distances.append(segment_to_segment_distance(start, end, edge_start, edge_end))

        return min(distances)

    def _edges(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        corners = list(self.corners())
        return (
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        )
