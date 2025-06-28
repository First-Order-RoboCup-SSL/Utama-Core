from typing import Deque, List, Tuple
from entities.data.raw_vision import RawVisionData
from entities.game.game_frame import GameFrame
import time
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv

from run.refiners import PositionRefiner


class GameGater:

    @staticmethod
    def wait_until_game_valid(
        my_team_is_yellow: bool,
        my_team_is_right: bool,
        exp_friendly: int,
        exp_enemy: int,
        vision_buffers: List[Deque[RawVisionData]],
        position_refiner: PositionRefiner,
        is_pvp: bool,
        rsim_env: SSLBaseEnv = None,
    ) -> Tuple[GameFrame, GameFrame]:

        def _add_frame(
            my_game_frame: GameFrame, opp_game_frame: GameFrame
        ) -> Tuple[GameFrame, GameFrame]:
            if rsim_env:
                vision_frames = [rsim_env._frame_to_observations()[0]]
            else:
                vision_frames = [
                    buffer.popleft() if buffer else None for buffer in vision_buffers
                ]
            my_game_frame = position_refiner.refine(my_game_frame, vision_frames)
            if is_pvp:
                opp_game_frame = position_refiner.refine(opp_game_frame, vision_frames)

            return my_game_frame, opp_game_frame

        my_game_frame = GameFrame(0, my_team_is_yellow, my_team_is_right, {}, {}, None)

        if is_pvp:
            opp_game_frame = GameFrame(
                0, not my_team_is_yellow, not my_team_is_right, {}, {}, None
            )
        else:
            opp_game_frame = None

        my_game_frame, opp_game_frame = _add_frame(my_game_frame, opp_game_frame)

        while (
            len(my_game_frame.friendly_robots) < exp_friendly
            or len(my_game_frame.enemy_robots) < exp_enemy
            or my_game_frame.ball is None
        ):
            time.sleep(0.05)
            my_game_frame, opp_game_frame = _add_frame(my_game_frame, opp_game_frame)

        # assert that we don't see more robots than expected
        if len(my_game_frame.friendly_robots) > exp_friendly:
            raise ValueError(
                f"Too many friendly robots: {len(my_game_frame.friendly_robots)} > {exp_friendly}"
            )
        if len(my_game_frame.enemy_robots) > exp_enemy:
            raise ValueError(
                f"Too many enemy robots: {len(my_game_frame.enemy_robots)} > {exp_enemy}"
            )

        return my_game_frame, opp_game_frame
