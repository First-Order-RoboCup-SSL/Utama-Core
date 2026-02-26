import time
from typing import Deque, List, Optional, Tuple

from utama_core.data_processing.refiners import PositionRefiner
from utama_core.entities.data.raw_vision import RawVisionData
from utama_core.entities.game.game_frame import GameFrame
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv


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
        is_pvp: bool,
        rsim_env: SSLBaseEnv = None,
        wait_before_warn: float = 3.0,
    ) -> Tuple[GameFrame, Optional[GameFrame]]:
        """
        Waits until the game frame has the expected number of robots and a ball.
        This function continuously refines the game frame using vision data until the conditions are met.

        Returns:
            A tuple containing the refined game frame for the player's team and the opponent's team (if is_pvp is True).
        """

        def _add_frame(my_game_frame: GameFrame, opp_game_frame: GameFrame) -> Tuple[GameFrame, Optional[GameFrame]]:
            if rsim_env:
                vision_frames = [rsim_env._frame_to_observations()[0]]
                rsim_env.steps += 1  # Increment the step count to simulate time passing in the environment
            else:
                vision_frames = [buffer.popleft() if buffer else None for buffer in vision_buffers]
            my_game_frame = position_refiner.refine(my_game_frame, vision_frames)
            if is_pvp:
                opp_game_frame = position_refiner.refine(opp_game_frame, vision_frames)

            return my_game_frame, opp_game_frame

        start_time = time.time()

        my_game_frame = GameFrame(0, my_team_is_yellow, my_team_is_right, {}, {}, None)

        if is_pvp:
            opp_game_frame = GameFrame(0, not my_team_is_yellow, not my_team_is_right, {}, {}, None)
        else:
            opp_game_frame = None

        my_game_frame, opp_game_frame = _add_frame(my_game_frame, opp_game_frame)

        while (
            len(my_game_frame.friendly_robots) < exp_friendly
            or len(my_game_frame.enemy_robots) < exp_enemy
            or (my_game_frame.ball is None and exp_ball)
        ):
            if time.time() - start_time > wait_before_warn:
                start_time = time.time()
                print("Waiting for valid game frame...")
                print(f"Friendly robots: {len(my_game_frame.friendly_robots)}/{exp_friendly}")
                print(f"Enemy robots: {len(my_game_frame.enemy_robots)}/{exp_enemy}")
                print(f"Ball present: {my_game_frame.ball is not None} (exp: {exp_ball})\n")

                # nothing will change in rsim if we don't step it.
                # if no valid frame, likely misconfigured.
                if rsim_env:
                    raise TimeoutError(
                        f"Rsim environment did not produce a valid game frame after {wait_before_warn} seconds. Check the environment setup and vision data."
                    )
            time.sleep(0.05)
            my_game_frame, opp_game_frame = _add_frame(my_game_frame, opp_game_frame)

        # assert that we don't see more robots than expected
        if len(my_game_frame.friendly_robots) > exp_friendly:
            raise ValueError(f"Too many friendly robots: {len(my_game_frame.friendly_robots)} > {exp_friendly}")
        if len(my_game_frame.enemy_robots) > exp_enemy:
            raise ValueError(f"Too many enemy robots: {len(my_game_frame.enemy_robots)} > {exp_enemy}")

        return my_game_frame, opp_game_frame
