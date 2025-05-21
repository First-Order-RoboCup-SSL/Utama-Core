from typing import Deque, List
from entities.data.raw_vision import RawVisionData
from entities.game.game import Game
import time
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv

from refiners.position import PositionRefiner


class GameGater:

    @staticmethod
    def wait_until_game_valid(
        my_team_is_yellow: bool,
        my_team_is_right: bool,
        exp_friendly: int,
        exp_enemy: int,
        exp_ball: bool,
        vision_buffers: List[Deque[RawVisionData]],
        position_refiner: PositionRefiner,
        rsim_env: SSLBaseEnv = None,
    ) -> Game:
        game = Game(0, my_team_is_yellow, my_team_is_right, {}, {}, None)

        while (
            len(game.friendly_robots) < exp_friendly
            and len(game.enemy_robots) < exp_enemy
            and (exp_ball and game.ball is None)
        ):
            if rsim_env:
                vision_frames = [rsim_env._frame_to_observations()[0]]
            else:
                vision_frames = [
                    buffer.popleft() if buffer else None for buffer in vision_buffers
                ]
            game = position_refiner.refine(game, vision_frames)
            time.sleep(0.1)

        return game
