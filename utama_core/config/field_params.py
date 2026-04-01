from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class FieldBounds:
    top_left: tuple[float, float]
    bottom_right: tuple[float, float]

    @property
    def center(self) -> tuple[float, float]:
        """Calculates the geometric center of the field bounds."""
        cx = (self.top_left[0] + self.bottom_right[0]) / 2.0
        cy = (self.top_left[1] + self.bottom_right[1]) / 2.0
        return (cx, cy)


from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldDimensions:
    """Holds field dimensions and derives all geometric shapes."""

    full_field_half_length: float
    full_field_half_width: float
    half_defense_area_depth: float
    half_defense_area_width: float
    half_goal_width: float

    # --- Bounds ---

    @cached_property
    def full_field_bounds(self):
        return FieldBounds(
            top_left=(-self.full_field_half_length, self.full_field_half_width),
            bottom_right=(self.full_field_half_length, -self.full_field_half_width),
        )

    # --- Full field polygon ---

    @cached_property
    def full_field(self) -> np.ndarray:
        L = self.full_field_half_length
        W = self.full_field_half_width
        return np.array(
            [
                (L, W),
                (L, -W),
                (-L, -W),
                (-L, W),
            ]
        )

    # --- Goal lines ---

    @cached_property
    def right_goal_line(self) -> np.ndarray:
        L = self.full_field_half_length
        G = self.half_goal_width
        return np.array(
            [
                (L, G),
                (L, -G),
            ]
        )

    @cached_property
    def left_goal_line(self) -> np.ndarray:
        L = self.full_field_half_length
        G = self.half_goal_width
        return np.array(
            [
                (-L, G),
                (-L, -G),
            ]
        )

    # --- Defense areas ---

    @cached_property
    def right_defense_area(self) -> np.ndarray:
        L = self.full_field_half_length
        D = self.half_defense_area_depth
        W = self.half_defense_area_width
        return np.array(
            [
                (L, W),
                (L - 2 * D, W),
                (L - 2 * D, -W),
                (L, -W),
            ]
        )

    @cached_property
    def left_defense_area(self) -> np.ndarray:
        L = self.full_field_half_length
        D = self.half_defense_area_depth
        W = self.half_defense_area_width
        return np.array(
            [
                (-L, W),
                (-L + 2 * D, W),
                (-L + 2 * D, -W),
                (-L, -W),
            ]
        )


STANDARD_FIELD_DIMS = FieldDimensions(
    full_field_half_length=4.5,
    full_field_half_width=3.0,
    half_defense_area_depth=0.5,
    half_defense_area_width=1,
    half_goal_width=0.5,
)

GREAT_EXHIBITION_FIELD_DIMS = FieldDimensions(
    full_field_half_length=2.0,
    full_field_half_width=1.5,
    half_defense_area_depth=0.4,
    half_defense_area_width=0.8,
    half_goal_width=0.5,
)
