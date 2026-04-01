from dataclasses import dataclass

import numpy as np

from utama_core.config.field_params import FieldBounds, FieldDimensions


class Field:
    """Field class that contains all the information about the field.

    Call the class properties to get the field information
    """

    def __init__(
        self,
        my_team_is_right: bool,
        field_bounds: FieldBounds,
        field_dims: FieldDimensions,
    ):
        self.my_team_is_right = my_team_is_right

        self._field_bounds = field_bounds
        self._field_dims = field_dims

        self._half_length = (field_bounds.bottom_right[0] - field_bounds.top_left[0]) / 2
        self._half_width = (field_bounds.top_left[1] - field_bounds.bottom_right[1]) / 2

    @property
    def includes_left_goal(self) -> bool:
        return self._field_bounds.top_left[0] == -self._field_dims.full_field_half_length and (
            self._field_bounds.top_left[1] >= self._field_dims.half_goal_width
            and self._field_bounds.bottom_right[1] <= -self._field_dims.half_goal_width
        )

    @property
    def includes_right_goal(self) -> bool:
        return self._field_bounds.bottom_right[0] == self._field_dims.full_field_half_length and (
            self._field_bounds.top_left[1] >= self._field_dims.half_goal_width
            and self._field_bounds.bottom_right[1] <= -self._field_dims.half_goal_width
        )

    @property
    def includes_my_goal_line(self) -> bool:
        if self.my_team_is_right:
            return self.includes_right_goal
        else:
            return self.includes_left_goal

    @property
    def includes_opp_goal_line(self) -> bool:
        if self.my_team_is_right:
            return self.includes_left_goal
        else:
            return self.includes_right_goal

    @property
    def my_goal_line(self) -> np.ndarray:
        if self.my_team_is_right:
            return self._field_dims.right_goal_line
        else:
            return self._field_dims.left_goal_line

    @property
    def enemy_goal_line(self) -> np.ndarray:
        if self.my_team_is_right:
            return self._field_dims.left_goal_line
        else:
            return self._field_dims.right_goal_line

    @property
    def my_defense_area(self) -> np.ndarray:
        if self.my_team_is_right:
            return self._field_dims.right_defense_area
        else:
            return self._field_dims.left_defense_area

    @property
    def enemy_defense_area(self) -> np.ndarray:
        if self.my_team_is_right:
            return self._field_dims.left_defense_area
        else:
            return self._field_dims.right_defense_area

    @property
    def half_length(self) -> float:
        return self._half_length

    @property
    def half_width(self) -> float:
        return self._half_width

    @property
    def field_bounds(self) -> FieldBounds:
        return self._field_bounds

    @property
    def center(self) -> tuple[float, float]:
        return self._field_bounds.center

    ### Class Properties for standard field dimensions ###

    @property
    def full_field_half_length(self) -> float:
        return self._field_dims.full_field_half_length

    @property
    def full_field_half_width(self) -> float:
        return self._field_dims.full_field_half_width

    @property
    def half_goal_width(self) -> float:
        return self._field_dims.half_goal_width

    @property
    def left_goal_line(self) -> np.ndarray:
        return self._field_dims.left_goal_line

    @property
    def right_goal_line(self) -> np.ndarray:
        return self._field_dims.right_goal_line

    @property
    def left_defense_area(self) -> np.ndarray:
        return self._field_dims.left_defense_area

    @property
    def right_defense_area(self) -> np.ndarray:
        return self._field_dims.right_defense_area

    @property
    def full_field(self) -> np.ndarray:
        return self._field_dims.full_field

    @property
    def full_field_bounds(self) -> FieldBounds:
        return self._field_dims.full_field_bounds
