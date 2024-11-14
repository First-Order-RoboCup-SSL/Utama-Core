from entities.game.ball import Ball
from entities.game.robot import Robot


# a single entry of game state at this time step
class GameState:
    def __init__(
        self,
        ts: float,
        balls: list[Ball],
        robots: dict[list[Robot]],
    ):
        self._ts = ts
        self._balls = balls
        self._robots = robots

    @property
    def ts(self) -> float:
        return self._ts

    @property
    def balls(self) -> Ball:
        return self._balls

    @property
    def robots(self) -> list:
        return self._robots
