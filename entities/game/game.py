from entities.game.field import Field
from entities.game.gameState import GameState
from entities.data.vision import RobotData, BallData


class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(self):
        self._field = Field()
        self._game_state_history = []
        self._yellow_score = 0
        self._blue_score = 0

    @property
    def field(self) -> Field:
        return self._field

    @property
    def current_state(self) -> GameState:
        return self._game_state_history[-1] if self._game_state_history else None

    @property
    def game_state_history(self) -> list[GameState]:
        return self._game_state_history

    @property
    def yellow_score(self) -> int:
        return self._yellow_score

    @property
    def blue_score(self) -> int:
        return self._blue_score

    def add_state_from_vision(
        self,
        ts: float,
        yellow_robots_data: list[RobotData],
        blue_robots_data: list[RobotData],
        balls_data: list[BallData],
    ) -> None:
        new_state = GameState(
            ts, {"yellow": yellow_robots_data, "blue": blue_robots_data}, balls_data
        )
        self.update_state(new_state)

    def update_state(self, new_state: GameState) -> None:
        self._game_state_history.append(new_state)
