from entities.data.vision import RobotData, BallData


# a single entry of game state at this time step
class GameState:
    def __init__(
        self,
        ts: float,
        robots: dict[list[RobotData]],
        balls: list[BallData],
    ):
        self._ts = ts
        self._balls = balls
        self._robots = robots

    @property
    def ts(self) -> float:
        return self._ts

    @property
    def balls(self) -> list[BallData]:
        return self._balls

    @property
    def robots(self) -> dict[list[RobotData]]:
        return self._robots
