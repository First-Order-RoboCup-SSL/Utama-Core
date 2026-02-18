from dataclasses import dataclass

import numpy as np


class ClassProperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


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


class Field:
    """Field class that contains all the information about the field.

    Call the class properties to get the field information
    """

    # Class constants refer to the standard SSL field (9m x 6m)

    _HALF_GOAL_WIDTH = 0.5
    _HALF_DEFENSE_AREA_LENGTH = 0.5
    _HALF_DEFENSE_AREA_WIDTH = 1

    _FULL_FIELD_HALF_WIDTH = 3.0
    _FULL_FIELD_HALF_LENGTH = 4.5

    _RIGHT_GOAL_LINE = np.array(
        [
            (_FULL_FIELD_HALF_LENGTH, _HALF_GOAL_WIDTH),
            (_FULL_FIELD_HALF_LENGTH, -_HALF_GOAL_WIDTH),
        ]
    )

    _LEFT_GOAL_LINE = np.array(
        [
            (-_FULL_FIELD_HALF_LENGTH, _HALF_GOAL_WIDTH),
            (-_FULL_FIELD_HALF_LENGTH, -_HALF_GOAL_WIDTH),
        ]
    )

    _RIGHT_DEFENSE_AREA = np.array(
        [
            (_FULL_FIELD_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
            (
                _FULL_FIELD_HALF_LENGTH - 2 * _HALF_DEFENSE_AREA_LENGTH,
                _HALF_DEFENSE_AREA_WIDTH,
            ),
            (
                _FULL_FIELD_HALF_LENGTH - 2 * _HALF_DEFENSE_AREA_LENGTH,
                -_HALF_DEFENSE_AREA_WIDTH,
            ),
            (_FULL_FIELD_HALF_LENGTH, -_HALF_DEFENSE_AREA_WIDTH),
            (_FULL_FIELD_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
        ]
    )

    _LEFT_DEFENSE_AREA = np.array(
        [
            (-_FULL_FIELD_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
            (
                -_FULL_FIELD_HALF_LENGTH + 2 * _HALF_DEFENSE_AREA_LENGTH,
                _HALF_DEFENSE_AREA_WIDTH,
            ),
            (
                -_FULL_FIELD_HALF_LENGTH + 2 * _HALF_DEFENSE_AREA_LENGTH,
                -_HALF_DEFENSE_AREA_WIDTH,
            ),
            (-_FULL_FIELD_HALF_LENGTH, -_HALF_DEFENSE_AREA_WIDTH),
            (-_FULL_FIELD_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
        ]
    )

    _FULL_FIELD = np.array(
        [
            (-_FULL_FIELD_HALF_LENGTH, -_FULL_FIELD_HALF_WIDTH),
            (-_FULL_FIELD_HALF_LENGTH, _FULL_FIELD_HALF_WIDTH),
            (_FULL_FIELD_HALF_LENGTH, _FULL_FIELD_HALF_WIDTH),
            (_FULL_FIELD_HALF_LENGTH, -_FULL_FIELD_HALF_WIDTH),
        ]
    )

    def __init__(self, my_team_is_right: bool, field_bounds: FieldBounds):
        self.my_team_is_right = my_team_is_right

        self._field_bounds = field_bounds

        self._half_length = (field_bounds.bottom_right[0] - field_bounds.top_left[0]) / 2
        self._half_width = (field_bounds.top_left[1] - field_bounds.bottom_right[1]) / 2

    @property
    def includes_left_goal(self) -> bool:
        return self._field_bounds.top_left[0] == -self._FULL_FIELD_HALF_LENGTH and (
            self._field_bounds.top_left[1] >= self._HALF_GOAL_WIDTH
            and self._field_bounds.bottom_right[1] <= -self._HALF_GOAL_WIDTH
        )

    @property
    def includes_right_goal(self) -> bool:
        return self._field_bounds.bottom_right[0] == self._FULL_FIELD_HALF_LENGTH and (
            self._field_bounds.top_left[1] >= self._HALF_GOAL_WIDTH
            and self._field_bounds.bottom_right[1] <= -self._HALF_GOAL_WIDTH
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
            return self._RIGHT_GOAL_LINE
        else:
            return self._LEFT_GOAL_LINE

    @property
    def enemy_goal_line(self) -> np.ndarray:
        if self.my_team_is_right:
            return self._LEFT_GOAL_LINE
        else:
            return self._RIGHT_GOAL_LINE

    @property
    def my_defense_area(self) -> np.ndarray:
        if self.my_team_is_right:
            return self._RIGHT_DEFENSE_AREA
        else:
            return self._LEFT_DEFENSE_AREA

    @property
    def enemy_defense_area(self) -> np.ndarray:
        if self.my_team_is_right:
            return self._LEFT_DEFENSE_AREA
        else:
            return self._RIGHT_DEFENSE_AREA

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

    @ClassProperty
    def HALF_GOAL_WIDTH(cls) -> float:
        return cls._HALF_GOAL_WIDTH

    @ClassProperty
    def LEFT_GOAL_LINE(cls) -> np.ndarray:
        return cls._LEFT_GOAL_LINE

    @ClassProperty
    def RIGHT_GOAL_LINE(cls) -> np.ndarray:
        return cls._RIGHT_GOAL_LINE

    @ClassProperty
    def LEFT_DEFENSE_AREA(cls) -> np.ndarray:
        return cls._LEFT_DEFENSE_AREA

    @ClassProperty
    def RIGHT_DEFENSE_AREA(cls) -> np.ndarray:
        return cls._RIGHT_DEFENSE_AREA

    @ClassProperty
    def FULL_FIELD_HALF_LENGTH(cls) -> float:
        return cls._FULL_FIELD_HALF_LENGTH

    @ClassProperty
    def FULL_FIELD_HALF_WIDTH(cls) -> float:
        return cls._FULL_FIELD_HALF_WIDTH

    @ClassProperty
    def FULL_FIELD(cls) -> np.ndarray:
        return cls._FULL_FIELD

    @ClassProperty
    def FULL_FIELD_BOUNDS(cls) -> FieldBounds:
        return FieldBounds(
            top_left=(-cls._FULL_FIELD_HALF_LENGTH, cls._FULL_FIELD_HALF_WIDTH),
            bottom_right=(cls._FULL_FIELD_HALF_LENGTH, -cls._FULL_FIELD_HALF_WIDTH),
        )
