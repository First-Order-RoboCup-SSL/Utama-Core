from __future__ import annotations

import math
from typing import Dict, Union

from utama_core.entities.data.vector import Vector2D

Scalar = Union[float, int]
LimitedValue = Union[Scalar, Vector2D]


class AccelerationLimiter:
    """Limit the rate of change of scalar or 2D vector commands on a per-robot basis."""

    def __init__(self, max_acceleration: float, dt: float):
        if dt <= 0:
            raise ValueError("dt must be greater than zero")
        if max_acceleration < 0:
            raise ValueError("max_acceleration must be non-negative")
        self._max_acceleration = max_acceleration
        self._dt = dt
        self._last_values: Dict[int, LimitedValue] = {}

    def limit(self, robot_id: int, value: LimitedValue) -> LimitedValue:
        allowed_delta = self._max_acceleration * self._dt
        last_value = self._last_values.get(robot_id)

        if isinstance(value, Vector2D):
            limited_value = self._limit_vector(value, last_value, allowed_delta)
        else:
            limited_value = self._limit_scalar(float(value), last_value, allowed_delta)

        self._last_values[robot_id] = limited_value
        return limited_value

    def reset(self, robot_id: int) -> None:
        self._last_values.pop(robot_id, None)

    def _limit_scalar(
        self,
        value: float,
        last_value: LimitedValue | None,
        allowed_delta: float,
    ) -> float:
        previous = float(last_value) if isinstance(last_value, (int, float)) else 0.0
        delta = value - previous
        clamped_delta = max(-allowed_delta, min(allowed_delta, delta))
        return previous + clamped_delta

    def _limit_vector(
        self,
        value: Vector2D,
        last_value: LimitedValue | None,
        allowed_delta: float,
    ) -> Vector2D:
        if isinstance(last_value, Vector2D):
            previous = last_value
        elif isinstance(last_value, (int, float)):
            previous = Vector2D(float(last_value), float(last_value))
        else:
            previous = Vector2D(0.0, 0.0)

        delta_x = value.x - previous.x
        delta_y = value.y - previous.y
        delta_norm = math.hypot(delta_x, delta_y)

        if delta_norm <= allowed_delta or delta_norm == 0.0:
            return value

        scale = allowed_delta / delta_norm
        return Vector2D(previous.x + delta_x * scale, previous.y + delta_y * scale)
