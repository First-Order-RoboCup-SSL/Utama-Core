"""RefereeGeometry: configurable field dimensions for the CustomReferee."""

from dataclasses import dataclass

from utama_core.config.field_params import (
    STANDARD_FIELD_DIMS,
    FieldBounds,
    FieldDimensions,
)


@dataclass(frozen=True)
class RefereeGeometry:
    """Immutable field geometry used by referee rule checkers.

    All measurements are in metres, using the standard SSL coordinate system
    (origin at centre, +x toward right goal, +y toward top of field).
    """

    half_length: float
    half_width: float
    half_goal_width: float
    half_defense_length: float
    half_defense_width: float
    center_circle_radius: float

    @classmethod
    def from_standard_div_b(cls) -> "RefereeGeometry":
        """Return geometry matching the standard SSL Division B field."""
        return cls(
            half_length=STANDARD_FIELD_DIMS.full_field_half_length,  # 4.5
            half_width=STANDARD_FIELD_DIMS.full_field_half_width,  # 3.0
            half_goal_width=STANDARD_FIELD_DIMS.half_goal_width,  # 0.5
            half_defense_length=STANDARD_FIELD_DIMS.half_defense_area_depth,  # 0.5
            half_defense_width=STANDARD_FIELD_DIMS.half_defense_area_width,  # 1.0
            center_circle_radius=0.5,
        )

    @classmethod
    def from_field_dims(cls, field_dims: FieldDimensions, field_bounds: FieldBounds | None = None) -> "RefereeGeometry":
        """Build geometry from a FieldDimensions instance.

        If field_bounds is provided (e.g. a sub-field play area), half_length and
        half_width are derived from it; otherwise the full field extents are used.
        All goal/defense dimensions are taken from field_dims so that non-standard
        field sizes are fully supported.
        """
        bounds = field_bounds or field_dims.full_field_bounds
        half_length = (bounds.bottom_right[0] - bounds.top_left[0]) / 2.0
        half_width = (bounds.top_left[1] - bounds.bottom_right[1]) / 2.0
        return cls(
            half_length=half_length,
            half_width=half_width,
            half_goal_width=field_dims.half_goal_width,
            half_defense_length=field_dims.half_defense_area_depth,
            half_defense_width=field_dims.half_defense_area_width,
            center_circle_radius=0.5,
        )

    # ------------------------------------------------------------------
    # Spatial query helpers
    # ------------------------------------------------------------------

    def is_in_field(self, x: float, y: float) -> bool:
        """True if (x, y) is within the playing field (including boundary)."""
        return abs(x) <= self.half_length and abs(y) <= self.half_width

    def is_in_left_goal(self, x: float, y: float) -> bool:
        """True if the ball has crossed the left goal line inside the goal."""
        return x < -self.half_length and abs(y) < self.half_goal_width

    def is_in_right_goal(self, x: float, y: float) -> bool:
        """True if the ball has crossed the right goal line inside the goal."""
        return x > self.half_length and abs(y) < self.half_goal_width

    def is_in_left_defense_area(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the left defense area."""
        return x <= -self.half_length + 2 * self.half_defense_length and abs(y) <= self.half_defense_width

    def is_in_right_defense_area(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the right defense area."""
        return x >= self.half_length - 2 * self.half_defense_length and abs(y) <= self.half_defense_width
