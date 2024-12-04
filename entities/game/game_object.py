from abc import ABC
from enum import Enum

class Colour(Enum):
    YELLOW = 1
    BLUE = 2

class GameObject(ABC):
    pass

class Robot(GameObject):
    def __init__(self, colour: Colour, robot_id: int):
        self.id = robot_id
        self.colour = colour

class Ball(GameObject):
    pass
