import argparse
import pickle
import warnings
from typing import Generator, Union

from config.settings import REPLAY_BASE_PATH
from entities.game import Ball as GameBall
from entities.game import Game, GameFrame
from entities.game import Robot as GameRobot
from global_utils.mapping_utils import map_friendly_enemy_to_colors
from global_utils.math_utils import rad_to_deg
from replay.entities import ReplayMetadata
from rsoccer_simulator.src.Entities import Ball as RSoccerBall
from rsoccer_simulator.src.Entities import Frame as RSoccerFrame
from rsoccer_simulator.src.Entities import FrameSSL
from rsoccer_simulator.src.Entities import Robot as RSoccerRobot
from rsoccer_simulator.src.ssl.envs import SSLStandardEnv


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


def play_replay(file_name: str):
    replay_path = REPLAY_BASE_PATH / f"{file_name}.pkl"
    replay_iter = _load_replay(replay_path)
    metadata = next(replay_iter)
    n_yellow, n_blue = map_friendly_enemy_to_colors(
        metadata.my_team_is_yellow,
        metadata.exp_friendly,
        metadata.exp_enemy,
    )
    replay_env = ReplayStandardSSL(n_robots_yellow=n_yellow, n_robots_blue=n_blue)

    for frame in replay_iter:
        if not isinstance(frame, GameFrame):
            warnings.warn(f"Invalid frame in replay file (type: {type(frame).__name__}), skipping.")
            continue
        replay_env.step_replay(frame)


def main():
    parser = argparse.ArgumentParser(description="Read and play a replay file.")
    parser.add_argument(
        "--replay-file",
        type=str,
        required=True,
        help="The name of the replay file (without extension) stored in ./replay/replays folder.",
    )
    args = parser.parse_args()

    play_replay(args.replay_file)


if __name__ == "__main__":
    main()
