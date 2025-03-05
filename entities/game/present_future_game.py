from entities.game.game import Game
from entities.game.past_game import PastGame


class PresentFutureGame():
    def __init__(self, past: PastGame, current: Game):
        self.__past = past
        self.current = current

    def add_game(self, game: Game):
        self.__past.add_game(self.current)
        self.current = game

    # def predict_/