from abc import ABC, abstractmethod
from typing import Any

from entities.game.game import Game




class BaseRefiner(ABC):
    
    @abstractmethod
    def refine(self, game: Game, data: Any) -> Game:
        ...