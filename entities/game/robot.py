from typing import Tuple, Union, Optional, List
import numpy as np

from entities.data.vision import RobotData
from entities.game.role import Attack, Defend, Role

ROLES: List[Role] = [Attack(), Defend()]

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
        return self._robot_data

    @robot_data.setter
    def robot_data(self, robot_data: RobotData):
        self._robot_data = robot_data
    
    @property
    def x(self) -> float:
        return self._robot_data[0]

    @property
    def y(self) -> float:
        return self._robot_data[1]

    @property
    def orentation(self) -> float:
        return self._robot_data[2]
        
    @property
    def inactive(self) -> bool:
        return self._inactive
    
    @inactive.setter
    def inactive(self, value: bool):
        self._inactive = value

class Friendly(Robot):
    def __init__(self, robot_id: int, role_id: Optional[int] = None, robot_data: Optional[RobotData] = None):
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
    def role(self, input: Union[int, str]):
        # TODO: docstring
        if isinstance(input, int):
            if 0 <= input < len(ROLES):
                self._role = ROLES[input]
            else:
                raise ValueError(f"Invalid role index: {input}")
        elif isinstance(input, str):
            for _role in ROLES:
                if _role.name == input:
                    self._role = _role
        else:
            raise ValueError(f"Invalid role: {input}")    
    
    @property
    def aggro_rating(self) -> float:
        return self._aggro_rating
    
    @aggro_rating.setter
    def aggro_rating(self, input: float):
        if input >= 0 :
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
        
    robot.role = "defender" # or 1 changes the role of the robot
    print(f"Role after change: {robot.role.name}")
    
    robot.aggro_rating = 5
    print(robot.aggro_rating)
    
    print(f"robot data before: {robot.robot_data}")
    robot.robot_data = robot_data
    print(f"robot data after: {robot.robot_data}")
    
    print(f"robot data before: {robot.has_ball}")
    robot.has_ball = True
    print(f"robot data after: {robot.has_ball}")
