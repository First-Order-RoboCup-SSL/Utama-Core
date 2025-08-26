import numpy as np
from shapely import Polygon, LineString
from shapely.geometry import Point


class ClassProperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


class Field:
    """
    Field class that contains all the information about the field

    Call the class properties to get the field information
    """

    _HALF_LENGTH = 4.5  # x value
    _HALF_WIDTH = 3  # y value
    _HALF_GOAL_WIDTH = 0.5 - 0.01 # slightly inside the posts (goal post thickness)
    _HALF_DEFENSE_AREA_LENGTH = 0.5
    _HALF_DEFENSE_AREA_WIDTH = 1
    _RIGHT_GOAL_LINE = LineString(
        [
            (_HALF_LENGTH, _HALF_GOAL_WIDTH),
            (_HALF_LENGTH, -_HALF_GOAL_WIDTH),
        ]
    )
    _LEFT_GOAL_LINE = LineString(
        [
            (-_HALF_LENGTH, _HALF_GOAL_WIDTH),
            (-_HALF_LENGTH, -_HALF_GOAL_WIDTH),
        ]
    )
    _CENTER_CIRCLE = Point(0, 0).buffer(0.5)  # center circle with radius 500
    _RIGHT_DEFENSE_AREA = Polygon(
        [
            (_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
            (
                _HALF_LENGTH - 2 * _HALF_DEFENSE_AREA_LENGTH,
                _HALF_DEFENSE_AREA_WIDTH,
            ),
            (
                _HALF_LENGTH - 2 * _HALF_DEFENSE_AREA_LENGTH,
                -_HALF_DEFENSE_AREA_WIDTH,
            ),
            (_HALF_LENGTH, -_HALF_DEFENSE_AREA_WIDTH),
            (_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
        ]
    )
    _LEFT_DEFENSE_AREA = Polygon(
        [
            (-_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
            (
                -_HALF_LENGTH + 2 * _HALF_DEFENSE_AREA_LENGTH,
                _HALF_DEFENSE_AREA_WIDTH,
            ),
            (
                -_HALF_LENGTH + 2 * _HALF_DEFENSE_AREA_LENGTH,
                -_HALF_DEFENSE_AREA_WIDTH,
            ),
            (-_HALF_LENGTH, -_HALF_DEFENSE_AREA_WIDTH),
            (-_HALF_LENGTH, _HALF_DEFENSE_AREA_WIDTH),
        ]
    )
    _FULL_FIELD = Polygon(
        [
            [-_HALF_LENGTH, -_HALF_WIDTH],
            [-_HALF_LENGTH, _HALF_WIDTH],
            [_HALF_LENGTH, _HALF_WIDTH],
            [_HALF_LENGTH, -_HALF_WIDTH],
        ]
    )

    def __init__(self, my_team_is_right: bool):
        self.my_team_is_right = my_team_is_right

    @property
    def my_goal_line(self) -> LineString:
        if self.my_team_is_right:
            return self._RIGHT_GOAL_LINE
        else:
            return self._LEFT_GOAL_LINE

    @property
    def enemy_goal_line(self) -> LineString:
        if self.my_team_is_right:
            return self._LEFT_GOAL_LINE
        else:
            return self._RIGHT_GOAL_LINE

    @property
    def my_defense_area(self) -> LineString:
        if self.my_team_is_right:
            return self._RIGHT_DEFENSE_AREA
        else:
            return self._LEFT_DEFENSE_AREA

    @property
    def enemy_defense_area(self) -> LineString:
        if self.my_team_is_right:
            return self._LEFT_DEFENSE_AREA
        else:
            return self._RIGHT_DEFENSE_AREA

    @ClassProperty
    def half_length(cls) -> float:
        return cls._HALF_LENGTH

    @ClassProperty
    def half_width(cls) -> float:
        return cls._HALF_WIDTH

    @ClassProperty
    def half_goal_width(cls) -> float:
        return cls._HALF_GOAL_WIDTH

    @ClassProperty
    def left_goal_line(cls) -> LineString:
        return cls._LEFT_GOAL_LINE

    @ClassProperty
    def right_goal_line(cls) -> LineString:
        return cls._RIGHT_GOAL_LINE

    @ClassProperty
    def center_circle(cls) -> Point:
        return cls._CENTER_CIRCLE

    @ClassProperty
    def left_defense_area(cls) -> Polygon:
        return cls._LEFT_DEFENSE_AREA

    @ClassProperty
    def right_defense_area(cls) -> Polygon:
        return cls._RIGHT_DEFENSE_AREA

    @ClassProperty
    def full_field(cls) -> Polygon:
        return cls._FULL_FIELD
