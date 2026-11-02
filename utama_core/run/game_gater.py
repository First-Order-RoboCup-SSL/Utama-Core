import time
from typing import Deque, Dict, List, Optional, Tuple

from utama_core.data_processing.refiners import PositionRefiner
from utama_core.entities.data.raw_vision import RawVisionData
from utama_core.entities.game.game_frame import GameFrame
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv


class GameGater:
    @staticmethod
    def _validate_mapping_covers_game_frame(mapping: Dict[int, int], observed_ids: set, team_label: str) -> None:
        """Verify that mapping keys exactly match observed vision IDs in the first game frame.

        Raises ValueError if any robot visible in the frame has no mapping entry,
        or if the mapping names IDs that were not observed.
        """
        if not mapping:
            return
        missing = observed_ids - mapping.keys()
        extra = mapping.keys() - observed_ids
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing entries for observed IDs {sorted(missing)}")
            if extra:
                parts.append(f"extra entries for unseen IDs {sorted(extra)}")
            raise ValueError(
                f"vision_to_cmd_mapping for {team_label} team does not match observed vision IDs: "
                + "; ".join(parts)
                + f". Observed IDs: {sorted(observed_ids)}."
            )

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
        my_vision_to_cmd_mapping: Optional[Dict[int, int]] = None,
        opp_vision_to_cmd_mapping: Optional[Dict[int, int]] = None,
    ) -> Tuple[GameFrame, Optional[GameFrame]]:
        """
        Waits until the game frame has the expected number of robots and a ball.
        This function continuously refines the game frame using vision data until the conditions are met.

        Returns:
            A tuple containing the refined game frame for the player's team and the opponent's team (if is_pvp is True).
        """

        def vision_summary(game_frame: GameFrame) -> str:
            """One line: what vision sees against what the run expects, with the mapped IDs."""

            def side(name, robots, exp, mapping):
                want = f" of {sorted(mapping)}" if mapping else ""
                return f"{name} {len(robots)}/{exp} {sorted(robots)}{want}"

            ball = "ball seen" if game_frame.ball is not None else ("no ball" if exp_ball else "ball not needed")
            return "Waiting for vision: " + ", ".join(
                (
                    side("ours", game_frame.friendly_robots, exp_friendly, my_vision_to_cmd_mapping),
                    side("theirs", game_frame.enemy_robots, exp_enemy, opp_vision_to_cmd_mapping),
                    ball,
                )
            )

        def print_current_vision(game_frame: GameFrame):
            print(vision_summary(game_frame))

        def _add_frame(my_game_frame: GameFrame, opp_game_frame: GameFrame) -> Tuple[GameFrame, Optional[GameFrame]]:
            if rsim_env:
                obs = rsim_env.step_noop()  # Step the environment without action to get the latest observation
                vision_frames = [obs]
            else:
                vision_frames = [buffer.popleft() if buffer else None for buffer in vision_buffers]
            my_game_frame = position_refiner.refine(
                my_game_frame,
                vision_frames,
            )
            if is_pvp:
                opp_game_frame = position_refiner.refine(
                    opp_game_frame,
                    vision_frames,
                )

            return my_game_frame, opp_game_frame

        start_time = time.time()

        my_game_frame = GameFrame(0, my_team_is_yellow, my_team_is_right, {}, {}, None)

        if is_pvp:
            opp_game_frame = GameFrame(0, not my_team_is_yellow, not my_team_is_right, {}, {}, None)
        else:
            opp_game_frame = None

        my_game_frame, opp_game_frame = _add_frame(my_game_frame, opp_game_frame)

        last_printed = None
        while (
            len(my_game_frame.friendly_robots) < exp_friendly
            or len(my_game_frame.enemy_robots) < exp_enemy
            or (my_game_frame.ball is None and exp_ball)
        ):
            # After `wait_before_warn`, a line whenever what vision sees changes, not a block every
            # few seconds: a field-day start printed screens of the same five lines.
            summary = vision_summary(my_game_frame)
            if time.time() - start_time > wait_before_warn and summary != last_printed:
                last_printed = summary
                print(summary)
                # nothing will change in rsim if we don't step it.
                # if no valid frame, likely misconfigured.
            if rsim_env:
                print_current_vision(my_game_frame)
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
