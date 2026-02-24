"""RefereeGeometry: configurable field dimensions for the CustomReferee."""

from dataclasses import dataclass

from utama_core.entities.game.field import Field, FieldBounds


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
            half_length=Field._FULL_FIELD_HALF_LENGTH,  # 4.5
            half_width=Field._FULL_FIELD_HALF_WIDTH,  # 3.0
            half_goal_width=Field._HALF_GOAL_WIDTH,  # 0.5
            half_defense_length=Field._HALF_DEFENSE_AREA_LENGTH,  # 0.5
            half_defense_width=Field._HALF_DEFENSE_AREA_WIDTH,  # 1.0
            center_circle_radius=0.5,
        )

    @classmethod
    def from_field_bounds(cls, field_bounds: FieldBounds) -> "RefereeGeometry":
        """Derive half_length/width from a FieldBounds; use Field constants for the rest."""
        half_length = (field_bounds.bottom_right[0] - field_bounds.top_left[0]) / 2.0
        half_width = (field_bounds.top_left[1] - field_bounds.bottom_right[1]) / 2.0
        return cls(
            half_length=half_length,
            half_width=half_width,
            half_goal_width=Field._HALF_GOAL_WIDTH,
            half_defense_length=Field._HALF_DEFENSE_AREA_LENGTH,
            half_defense_width=Field._HALF_DEFENSE_AREA_WIDTH,
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
