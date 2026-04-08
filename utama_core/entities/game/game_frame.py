import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field
from utama_core.entities.game.robot import Robot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameFrame:
    ts: float
    my_team_is_yellow: bool
    my_team_is_right: bool
    friendly_robots: Dict[int, Robot]
    enemy_robots: Dict[int, Robot]
    ball: Optional[Ball]
