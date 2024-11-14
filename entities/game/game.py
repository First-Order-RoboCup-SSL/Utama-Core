from entities.game.field import Field
from entities.game.ball import Ball


class Game:
    def __init__(self):
        self._field = Field()
        self._balls, self._robots = self.update_game()

    def update_game(self, balls, robots):
        self._balls = balls
        self._robots = robots

    @property
    def field(self) -> Field:
        return self._field

    @property
    def ball(self) -> Ball:
        return self._ball

    @property
    def robots(self) -> list:
        return self._robots
