from typing import Tuple
from dataclasses import dataclass
import logging

from entities.data.vision import RobotData
from __future__ import annotations

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Robot:
    id: int 
    is_friendly: bool
    has_ball: bool # Friendly and enemy now have this, friendly is from IR sensor, enemy from vision? 
    x: float
    y: float 
    orientation: float

    def coords(self) -> Tuple[float, float]:
        return self.x, self.y