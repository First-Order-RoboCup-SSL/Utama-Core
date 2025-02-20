import numpy as np
from shapely import Polygon, LineString
from shapely.geometry import Point


class Field:
    def __init__(self, my_team_is_yellow: bool, my_team_right: bool):  # TODO: Make these static
        self.my_team_right = my_team_right
        self.my_team_is_yellow = my_team_is_yellow
        
        self.HALF_LENGTH = 4.5  # x value
        self.HALF_WIDTH = 3  # y value
        self.HALF_GOAL_WIDTH = 0.5
        self.HALF_DEFENSE_AREA_LENGTH = 0.5
        self.HALF_DEFENSE_AREA_WIDTH = 1
        self.RIGHT_GOAL_LINE = LineString(
            [
                (self.HALF_LENGTH, self.HALF_GOAL_WIDTH),
                (self.HALF_LENGTH, -self.HALF_GOAL_WIDTH),
            ]
        )
        self.LEFT_GOAL_LINE = LineString(
            [
                (-self.HALF_LENGTH, self.HALF_GOAL_WIDTH),
                (-self.HALF_LENGTH, -self.HALF_GOAL_WIDTH),
            ]
        )
        self.CENTER_CIRCLE = Point(0, 0).buffer(0.5)  # center circle with radius 500
        self.RIGHT_DEFENSE_AREA = Polygon(
            [
                (self.HALF_LENGTH, self.HALF_DEFENSE_AREA_WIDTH),
                (
                    self.HALF_LENGTH - 2 * self.HALF_DEFENSE_AREA_LENGTH,
                    self.HALF_DEFENSE_AREA_WIDTH,
                ),
                (
                    self.HALF_LENGTH - 2 * self.HALF_DEFENSE_AREA_LENGTH,
                    -self.HALF_DEFENSE_AREA_WIDTH,
                ),
                (self.HALF_LENGTH, -self.HALF_DEFENSE_AREA_WIDTH),
                (self.HALF_LENGTH, self.HALF_DEFENSE_AREA_WIDTH),
            ]
        )
        self.LEFT_DEFENSE_AREA = Polygon(
            [
                (-self.HALF_LENGTH, self.HALF_DEFENSE_AREA_WIDTH),
                (
                    -self.HALF_LENGTH + 2 * self.HALF_DEFENSE_AREA_LENGTH,
                    self.HALF_DEFENSE_AREA_WIDTH,
                ),
                (
                    -self.HALF_LENGTH + 2 * self.HALF_DEFENSE_AREA_LENGTH,
                    -self.HALF_DEFENSE_AREA_WIDTH,
                ),
                (-self.HALF_LENGTH, -self.HALF_DEFENSE_AREA_WIDTH),
                (-self.HALF_LENGTH, self.HALF_DEFENSE_AREA_WIDTH),
            ]
        )

    def my_goal_line(self, my_team_is_yellow) -> LineString:
        if my_team_is_yellow:
            return self.RIGHT_GOAL_LINE
        else:
            return self.LEFT_GOAL_LINE

    def enemy_goal_line(self, my_team_is_yellow) -> LineString:
        if my_team_is_yellow:
            return self.LEFT_GOAL_LINE
        else:
            return self.RIGHT_GOAL_LINE

    def my_defense_area(self, my_team_is_yellow) -> LineString:
        if my_team_is_yellow:
            return self.yellow_defense_area
        else:
            return self.blue_defense_area

    def enemy_defense_area(self, my_team_is_yellow) -> LineString:
        if my_team_is_yellow:
            return self.blue_defense_area
        else:
            return self.yellow_defense_area

    @property
    def half_length(self) -> float:
        return self.HALF_LENGTH

    @property
    def half_width(self) -> float:
        return self.HALF_WIDTH

    @property
    def half_goal_width(self) -> float:
        return self.HALF_GOAL_WIDTH

    @property
    def yellow_goal_line(self) -> LineString:
        return self.YELLOW_GOAL_LINE

    @property
    def blue_goal_line(self) -> LineString:
        return self.BLUE_GOAL_LINE

    @property
    def center_circle(self) -> Point:
        return self.CENTER_CIRCLE

    @staticmethod
    def yellow_defense_area() -> Polygon:
        return Field().YELLOW_DEFENSE_AREA

    @staticmethod
    def blue_defense_area() -> Polygon:
        return Field().BLUE_DEFENSE_AREA

    @staticmethod
    def full_field() -> Polygon:
        return Polygon(
            [
                [-Field.HALF_LENGTH, -Field.HALF_WIDTH],
                [-Field.HALF_LENGTH, Field.HALF_WIDTH],
                [Field.HALF_LENGTH, Field.HALF_WIDTH],
                [Field.HALF_LENGTH, -Field.HALF_WIDTH],
            ]
        )
