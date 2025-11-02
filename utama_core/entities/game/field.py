from dataclasses import dataclass
from typing import Tuple

from shapely import LineString, Polygon
from shapely.geometry import Point


class ClassProperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


@dataclass(frozen=True)
class FieldConfig:
    # two opposite corners of an axis-aligned rectangle
    corner1: Tuple[float, float]
    corner2: Tuple[float, float]

    half_goal_width: float = 0.5
    half_def_area_length: float = 0.5
    half_def_area_width: float = 1.0
    center_circle_radius: float = 0.5


class Field:
    """Field class that contains all the information about the field.

    Call the class properties to get the field information
    """

    _HALF_GOAL_WIDTH = 0.5
    _HALF_DEFENSE_AREA_LENGTH = 0.5
    _HALF_DEFENSE_AREA_WIDTH = 1

    _CENTER_CIRCLE = Point(0, 0).buffer(0.5)  # center circle with radius 500

    def __init__(self, my_team_is_right: bool, field_config: FieldConfig):
        self.my_team_is_right = my_team_is_right
        self.field_config = field_config

        # normalize the field to be centered at (0,0)
        (x1, y1), (x2, y2) = field_config[0], field_config[1]
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

        width_x = x_max - x_min
        width_y = y_max - y_min

        self._half_length = width_x / 2.0
        self._half_width = width_y / 2.0
        self._half_goal_width = 0.5
        self._half_def_len = 0.5
        self._half_def_wid = 1.0

        # build geometry in the centered frame
        HL = self._half_length
        HW = self._half_width
        HGW = self._half_goal_width
        HDL = self._half_def_len
        HDW = self._half_def_wid

        self._right_goal_line = LineString([(HL, HGW), (HL, -HGW)])
        self._left_goal_line = LineString([(-HL, HGW), (-HL, -HGW)])
        self._center_circle = Point(0.0, 0.0).buffer(0.5)

        self._right_defense_area = Polygon(
            [
                (HL, HDW),
                (HL - 2 * HDL, HDW),
                (HL - 2 * HDL, -HDW),
                (HL, -HDW),
                (HL, HDW),
            ]
        )
        self._left_defense_area = Polygon(
            [
                (-HL, HDW),
                (-HL + 2 * HDL, HDW),
                (-HL + 2 * HDL, -HDW),
                (-HL, -HDW),
                (-HL, HDW),
            ]
        )
        self._full_field = Polygon(
            [
                (-HL, -HW),
                (-HL, HW),
                (HL, HW),
                (HL, -HW),
            ]
        )

    @property
    def my_goal_line(self) -> LineString:
        if self.my_team_is_right:
            return self.right_goal_line
        else:
            return self.left_goal_line

    @property
    def enemy_goal_line(self) -> LineString:
        if self.my_team_is_right:
            return self.left_goal_line
        else:
            return self.right_goal_line

    @property
    def my_defense_area(self) -> LineString:
        if self.my_team_is_right:
            return self.right_defense_area
        else:
            return self.left_defense_area

    @property
    def enemy_defense_area(self) -> LineString:
        if self.my_team_is_right:
            return self.left_defense_area
        else:
            return self.right_defense_area

    @property
    def half_length(self) -> float:
        return self.half_length

    @property
    def half_width(self) -> float:
        return self.half_width

    @ClassProperty
    def half_goal_width(cls) -> float:
        return cls.HALF_GOAL_WIDTH

    @property
    def left_goal_line(self) -> LineString:
        return self.left_goal_line

    @property
    def right_goal_line(self) -> LineString:
        return self.right_goal_line

    @ClassProperty
    def center_circle(cls) -> Point:
        return cls._CENTER_CIRCLE

    @property
    def left_defense_area(self) -> Polygon:
        return self.left_defense_area

    @property
    def right_defense_area(self) -> Polygon:
        return self.right_defense_area

    @property
    def full_field(self) -> Polygon:
        return self.full_field
