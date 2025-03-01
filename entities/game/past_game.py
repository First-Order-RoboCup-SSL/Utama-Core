import copy

from entities.game.game import Game

class PastGame:
    def __init__(self, max_history: int):
        self.max_history = max_history
        self.games = []

    def add_game(self, game: Game):
        self.games.append(copy.deepcopy(game))
        if len(self.games) > self.max_history:
            self.games.pop(0)

    def n_steps_ago(self, n):
        return self.games[-n]