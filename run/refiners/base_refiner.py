from abc import ABC, abstractmethod
from typing import Any

from entities.game.game_frame import GameFrame


class BaseRefiner(ABC):
    @abstractmethod
    def refine(self, game: GameFrame, data: Any) -> GameFrame: ...
