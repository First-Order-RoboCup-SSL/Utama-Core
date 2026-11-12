"""Scale a point authored for the standard field to the current field size.

Ported from `utama_strategy.utils.field_scaling` — depended only on Core
internals already, so it moves here rather than staying duplicated.
"""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import Field


def scale_point_from_standard_field(point: Vector2D, field: Field) -> Vector2D:
    return Vector2D(
        point.x * field.half_length / STANDARD_FIELD_DIMS.full_field_half_length,
        point.y * field.half_width / STANDARD_FIELD_DIMS.full_field_half_width,
    )
