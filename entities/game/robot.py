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

    def __bool__(self):
        return self._robot_data is not None
    
    def __repr__(self):
        return f"Robot(id={self.id}, x={self.x}, y={self.y}, orientation={self.orientation})"
    
    @property
    def id(self) -> int:
        return self._id

    @property
    def robot_data(self) -> RobotData:
        if self._robot_data is not None:
            return self._robot_data
        elif self.inactive:
            logger.critical(f" Should not be getting coords of robot_id: {self.id} (inactive)")
            return None
        else:
            logger.warning(f" Should not be getting coords of robot_id: {self.id} (None)")
            return None

    @robot_data.setter
    def robot_data(self, robot_data: RobotData):
        self._robot_data = robot_data

    @property
    def x(self) -> float:
        if self._robot_data is not None:
            return self._robot_data[0]
        elif self.inactive:
            logger.critical(f" Should not be getting coords of robot_id: {self.id} (inactive)")
            return None
        else:
            logger.critical(f" Should not be getting coords of robot_id: {self.id} (None)")
            return None

    @property
    def y(self) -> float:
        if self._robot_data is not None:
            return self._robot_data[1]
        elif self.inactive:
            logger.critical(f" Should not be getting coords of robot_id: {self.id} (inactive)")
            return None
        else:
            logger.critical(f" Should not be getting coords of robot_id: {self.id} (None)")
            return None

    @property
    def orientation(self) -> float:
        if self._robot_data is not None:
            return self._robot_data[2]
        elif self.inactive:
            logger.critical(f" Should not be getting coords of robot_id: {self.id} (inactive)")
            return None
        else:
            logger.critical(f" Should not be getting coords of this robot_id: {self.id} (None)")
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
