from typing import Tuple, Union, Optional, List
import numpy as np

from entities.data.vision import RobotData
from entities.game.role import Attack, Defend, Role

from enum import Enum


class RoleType(Enum):
    ATTACK = 1
    DEFEND = 2


import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Robot:
    def __init__(self, robot_id: int, robot_data: Optional[RobotData] = None):
        self._id = robot_id
        self._robot_data = robot_data
        self._inactive = False

    @property
    def id(self) -> int:
        return self._id

    @property
    def robot_data(self) -> RobotData:
        if self.inactive:
            return self._robot_data
        else:
            logger.warning(" Should not be getting coords of this robot (inactive)")
            return None

    @robot_data.setter
    def robot_data(self, robot_data: RobotData):
        self._robot_data = robot_data

    @property
    def x(self) -> float:
        if not self.inactive:
            return self._robot_data[0]
        else:
            logger.warning(" Should not be getting x-coords of this robot (inactive)")
            return None

    @property
    def y(self) -> float:
        if not self.inactive:
            return self._robot_data[1]
        else:
            logger.warning(" Should not be getting y-coords of this robot (inactive)")
            return None

    @property
    def orientation(self) -> float:
        if not self.inactive:
            return self._robot_data[2]
        else:
            logger.warning(
                " Should not be getting orientation data of this robot (inactive)"
            )
            return None

    @property
    def inactive(self) -> bool:
        return self._inactive

    @inactive.setter
    def inactive(self, value: bool):
        self._inactive = value


class Friendly(Robot):
    def __init__(
        self,
        robot_id: int,
        role_id: Optional[int] = None,
        robot_data: Optional[RobotData] = None,
    ):
        super().__init__(robot_id, robot_data)
        self._has_ball: bool = False
        self._aggro_rating: float = 0
        self._sprt_rbt_ids: List[int] = None
        self._role = None

        if role_id != None and self._role == None:
            self.role = role_id

    @property
    def has_ball(self) -> bool:
        return self._has_ball

    @has_ball.setter
    def has_ball(self, value: bool):
        self._has_ball = value

    @property
    def sprt_rbt_ids(self) -> List[int]:
        return self._sprt_rbt_ids

    @sprt_rbt_ids.setter
    def sprt_rbt_ids(self, sprt_rbt_ids: List[int]):
        self._sprt_rbt_ids = list(set(sprt_rbt_ids))  # Ensure unique IDs

    @property
    def role(self) -> Role:
        return self._role

    @role.setter
    def role(self, input: RoleType):
        # TODO: docstring
        if RoleType.ATTACK:
            self._role = Attack()
        elif RoleType.DEFEND:
            self._role = Defend()

    @property
    def aggro_rating(self) -> float:
        return self._aggro_rating

    @aggro_rating.setter
    def aggro_rating(self, input: float):
        if input >= 0:
            self._aggro_rating = input
        else:
            raise ValueError("agro raiting value must be non-negative")


class Enemy(Robot):
    def __init__(self, robot_id: int, robot_data: Optional[RobotData] = None):
        super().__init__(robot_id, robot_data)
        # TODO: add properties like danger raiting etc


if __name__ == "__main__":
    robot_data = RobotData(0.5, 0.5, 0.5)
    robot = Friendly(0)

    if robot.role is None:
        print("Role before init: None")
    else:
        print(f"Role before init: {robot.role.name}")

    robot.role = "defender"  # or 1 changes the role of the robot
    print(f"Role after change: {robot.role.name}")

    robot.aggro_rating = 5
    print(robot.aggro_rating)

    print(f"robot data before: {robot.robot_data}")
    robot.robot_data = robot_data
    print(f"robot data after: {robot.robot_data}")

    print(f"robot data before: {robot.has_ball}")
    robot.has_ball = True
    print(f"robot data after: {robot.has_ball}")
