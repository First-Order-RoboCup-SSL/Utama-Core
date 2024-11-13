from entities.game.field import Field
from entities.game.ball import Ball


class Game:
    def __init__(self, my_team: str = "blue"):
        self.field = Field()
        self.ball = Ball()
        self.players = []
