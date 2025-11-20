import argparse
import logging
import pickle
import time
import warnings
from typing import Generator, Union

import pygame

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.entities.game import Ball as GameBall
from utama_core.entities.game import Game, GameFrame
from utama_core.entities.game import Robot as GameRobot
from utama_core.global_utils.mapping_utils import map_friendly_enemy_to_colors
from utama_core.global_utils.math_utils import rad_to_deg
from utama_core.replay.entities import ReplayMetadata
from utama_core.rsoccer_simulator.src.Entities import Ball as RSoccerBall
from utama_core.rsoccer_simulator.src.Entities import Frame as RSoccerFrame
from utama_core.rsoccer_simulator.src.Entities import FrameSSL
from utama_core.rsoccer_simulator.src.Entities import Robot as RSoccerRobot
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv

logger = logging.getLogger(__name__)


class ReplayStandardSSL(SSLStandardEnv):

    def _set_ball(self, game_ball: GameBall) -> RSoccerBall:
        rsoccer_ball = RSoccerBall()
        rsoccer_ball.x = game_ball.p.x
        rsoccer_ball.y = -game_ball.p.y
        rsoccer_ball.z = game_ball.p.z
        return rsoccer_ball

    def _set_robot(self, game_robot: GameRobot) -> RSoccerRobot:
        rsoccer_robot = RSoccerRobot()
        rsoccer_robot.id = game_robot.id
        rsoccer_robot.x = game_robot.p.x
        rsoccer_robot.y = -game_robot.p.y
        rsoccer_robot.theta = rad_to_deg(-game_robot.orientation)
        return rsoccer_robot

    def _set_frame(self, game_frame: GameFrame):
        rsoccer_frame = FrameSSL()
        rsoccer_frame.ball = self._set_ball(game_frame.ball)
        friendly_robot_dict: dict[int, RSoccerRobot] = {}
        enemy_robot_dict: dict[int, RSoccerRobot] = {}
        for robot in game_frame.friendly_robots.values():
            friendly_robot_dict[robot.id] = self._set_robot(robot)
        for robot in game_frame.enemy_robots.values():
            enemy_robot_dict[robot.id] = self._set_robot(robot)

        rsoccer_frame.robots_yellow, rsoccer_frame.robots_blue = map_friendly_enemy_to_colors(
            game_frame.my_team_is_yellow, friendly_robot_dict, enemy_robot_dict
        )

        return rsoccer_frame

    def step_replay(self, game_frame: GameFrame):
        self.frame = self._set_frame(game_frame)
        self.render()


def _load_replay(path) -> Generator[Union[ReplayMetadata, GameFrame], None, None]:
    """Generator that yields metadata and game frames from a replay file."""
    with open(path, "rb") as f:
        # read metadata (first object)
        metadata = pickle.load(f)

        # yield metadata separately
        yield metadata

        # then yield frames
        while True:
            try:
                frame = pickle.load(f)
                yield frame
            except EOFError:
                break


def play_replay(file_name: str, play_by_play: bool = False):
    replay_path = REPLAY_BASE_PATH / f"{file_name}.pkl"

    # Load all frames into memory
    frames = list(_load_replay(replay_path))
    if not frames:
        print("Replay file is empty!")
        return

    metadata = frames[0]
    game_frames = frames[1:]  # skip metadata
    n_yellow, n_blue = map_friendly_enemy_to_colors(
        metadata.my_team_is_yellow,
        metadata.exp_friendly,
        metadata.exp_enemy,
    )
    replay_env = ReplayStandardSSL(n_robots_yellow=n_yellow, n_robots_blue=n_blue)

    frame_index = 0

    while frame_index < len(game_frames):
        frame = game_frames[frame_index]

        if not isinstance(frame, GameFrame):
            warnings.warn(f"Invalid frame in replay file (type: {type(frame).__name__}), skipping.")
            frame_index += 1
            continue

        replay_env.step_replay(frame)

        if play_by_play:
            print(f"Frame {frame_index + 1}/{len(game_frames)}: Press RIGHT/SPACE to advance, LEFT to go back.")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                keys = pygame.key.get_pressed()
                step = 0

                # Forward step
                if keys[pygame.K_SPACE] or keys[pygame.K_RIGHT]:
                    step = 1
                # Backward step
                elif keys[pygame.K_LEFT]:
                    step = -1

                if step != 0:
                    frame_index = max(0, min(frame_index + step, len(game_frames) - 1))
                    waiting = False

                pygame.time.delay(10)
        else:
            frame_index += 1


def get_latest_replay_name() -> str:
    files = list(REPLAY_BASE_PATH.glob("*.pkl"))
    if not files:
        raise FileNotFoundError("No replay files found in the replay directory.")
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file.stem  # Return the file name without extension


def main():
    parser = argparse.ArgumentParser(description="Read and play a replay file.")
    parser.add_argument(
        "-n",
        "--replay-file",
        type=str,
        help="The name of the replay file (without extension) stored in ./replays folder.",
    )
    parser.add_argument(
        "-p",
        "--play-by-play",
        action="store_true",
        help="Render the replay one frame at a time for step-by-step playback.",
    )

    args = parser.parse_args()

    if args.replay_file:
        replay_file = args.replay_file
    else:
        replay_file = get_latest_replay_name()
        logger.info(f"No replay file specified. Using the latest replay: {replay_file}")

    play_replay(replay_file, play_by_play=args.play_by_play)


if __name__ == "__main__":
    main()
