## NOT CURRENTLY USED ###

from typing import Tuple


class Robot:
    def __init__(self, robot_id, team_colour):
        self._id = robot_id
        self._team_colour = team_colour
        self._has_ball = False
        self._pos = None
        self._heading = None
        self._records = []

    @property
    def id(self) -> int:
        return self._id

    @property
    def team_colour(self) -> str:
        return self._team_colour

    @property
    def has_ball(self) -> bool:
        return self._has_ball

    @property
    def pos(self) -> Tuple[float, float]:
        return self._pos

    @property
    def x(self) -> float:
        return self._pos[0]

    @property
    def y(self) -> float:
        return self._pos[1]

    @property
    def heading(self) -> float:
        return self._heading
