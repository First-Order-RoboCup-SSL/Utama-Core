from entities.game.field import Field
from entities.game.gameState import GameState


class Game:
    def __init__(self):
        self._field = Field()
        self._current_state = None
        self._game_state_history = []

    @property
    def field(self) -> Field:
        return self._field

    @property
    def current_state(self) -> GameState:
        return self._current_state

    @property
    def game_state_history(self) -> list[GameState]:
        return self._game_state_history

    def update_state(self, new_state: GameState):
        self._game_state_history[new_state.ts] = self._current_state
        self._current_state = new_state
