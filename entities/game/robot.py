from typing import Optional

from entities.data.vision import RobotData

import logging

# Configure logging
logger = logging.getLogger(__name__)


class Robot:
    def __init__(self, robot_id: int, is_friendly: bool, robot_data: Optional[RobotData] = None):
        self._id = robot_id
        self.is_friendly = is_friendly
        self._robot_data = robot_data
        self._inactive = False
        if is_friendly:
            self._has_ball = False

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
        
    @property
    def has_ball(self) -> bool:
        return self._has_ball

    @has_ball.setter
    def has_ball(self, value: bool):
        if self.is_friendly:
            self._has_ball = value
        else:
            raise AttributeError("Enemy robots cannot have the 'has_ball' property.")
        
    @has_ball.getter
    def has_ball(self) -> bool:
        if self.is_friendly:
            return self._has_ball
        else:
            raise AttributeError("Enemy robots cannot have the 'has_ball' property.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    
    robot_data_1 = RobotData(0.5, 0.5, 0.5)
    robot_data_2 = RobotData(0.2, 0.2, 0.2)
    
    ### game robot object ###
    
    game_friendly_robot = Robot(0, True, robot_data_1)
    game_enemy_robot = Robot(1, False, robot_data_2)
    
    print(f"Robot 1 coords: {game_friendly_robot.x}, {game_friendly_robot.y}")
    print(f"Robot 2 coords: {game_enemy_robot.x}, {game_enemy_robot.y}")
    
    game_friendly_robot.has_ball = True
    print(f"Robot 1 has ball: {game_friendly_robot.has_ball}")
    
    game_enemy_robot.has_ball = True  # raises an error
    print(f"Robot 2 has ball: {game_enemy_robot.has_ball}")
    
    
