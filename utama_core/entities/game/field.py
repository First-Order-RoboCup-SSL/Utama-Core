from dataclasses import dataclass

from shapely import LineString, Polygon
from shapely.geometry import Point


class ClassProperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


@dataclass(frozen=True)
class FieldConfig:
    top_left: tuple[float, float]
    bottom_right: tuple[float, float]


class Field:
    """Field class that contains all the information about the field.

    Call the class properties to get the field information
    """

    # Class constants refer to the standard SSL field (9m x 6m)

    _HALF_GOAL_WIDTH = 0.5
    _HALF_DEFENSE_AREA_LENGTH = 0.5
    _HALF_DEFENSE_AREA_WIDTH = 1

    _RIGHT_GOAL_LINE = LineString(
        [
            (4.5, 0.5),
            (4.5, -0.5),
        ]
    )

    _LEFT_GOAL_LINE = LineString(
        [
            (-4.5, 0.5),
            (-4.5, -0.5),
        ]
    )

    _CENTER_CIRCLE = Point(0.0, 0.0).buffer(0.5)
    _RIGHT_DEFENSE_AREA = Polygon(
        [
            (4.5, 1.0),
            (3.5, 1.0),
            (3.5, -1.0),
            (4.5, -1.0),
            (4.5, 1.0),
        ]
    )

    _LEFT_DEFENSE_AREA = Polygon(
        [
            (-4.5, 1.0),
            (-3.5, 1.0),
            (-3.5, -1.0),
            (-4.5, -1.0),
            (-4.5, 1.0),
        ]
    )

    _FULL_FIELD = Polygon(
        [
            (-4.5, -3.0),
            (-4.5, 3.0),
            (4.5, 3.0),
            (4.5, -3.0),
        ]
    )

    def __init__(self, my_team_is_right: bool, field_config: FieldConfig):
        self.my_team_is_right = my_team_is_right

        self._field_config = field_config

        self._half_length = (field_config.bottom_right[0] - field_config.top_left[0]) / 2
        self._half_width = (field_config.top_left[1] - field_config.bottom_right[1]) / 2

    @property
    def has_left_goal(self) -> bool:
        return self._field_config.top_left[0] == -4.5 and (
            self._field_config.top_left[1] >= 0.5 and self._field_config.bottom_right[1] <= -0.5
        )

    @property
    def has_right_goal(self) -> bool:
        return self._field_config.bottom_right[0] == 4.5 and (
            self._field_config.top_left[1] >= 0.5 and self._field_config.bottom_right[1] <= -0.5
        )

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

    @property
    def half_length(self) -> float:
        return self._half_length

    @property
    def half_width(self) -> float:
        return self._half_width

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
