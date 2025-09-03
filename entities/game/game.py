from entities.game.current_game_frame import CurrentGameFrame
from entities.game.game_frame import GameFrame
from entities.game.game_history import GameHistory


class Game:
    def __init__(self, past: GameHistory, current: GameFrame):
        self.__past = past
        self._current_game_standard_frame = current
        self.current = CurrentGameFrame(current)

    def add_game(self, game_frame: GameFrame):
        self.__past.add_game(self._current_game_standard_frame)
        self._current_game_standard_frame = game_frame
        self.current = CurrentGameFrame(game_frame)

    # def predict_/

    @property
    def my_team_is_yellow(self) -> bool:
        return self.current.my_team_is_yellow

    @property
    def my_team_is_right(self) -> bool:
        return self.current.my_team_is_right

    @property
    def friendly_robots(self):
        return self.current.friendly_robots

    @property
    def enemy_robots(self):
        return self.current.enemy_robots

    @property
    def ball(self):
        return self.current.ball

    @property
    def field(self):
        return self.current.field

    @property
    def robot_with_ball(self):
        return self.current.robot_with_ball

    @property
    def proximity_lookup(self):
        return self.current.proximity_lookup
