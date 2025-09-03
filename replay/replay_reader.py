import argparse
import pickle
from collections import deque

from config.settings import REPLAY_BASE_PATH
from entities.game import Game, GameFrame


def read_replay_file(file_name: str):
    replay_path = REPLAY_BASE_PATH / f"{file_name}.pkl"
    with open(replay_path, "rb") as f:
        replay_data = pickle.load(f)

    if not isinstance(replay_data, Game):
        raise ValueError("Not able to replay file, file is not a Game.")

    # name mangling
    history: deque[GameFrame] = replay_data._Game__past.raw_games_history
    history.append(replay_data.current)

    for game_frame in history:
        print(game_frame)


def main():
    parser = argparse.ArgumentParser(description="Read and play a replay file.")
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="The name of the replay file (without extension) stored in ./replay/replays folder.",
    )
    args = parser.parse_args()

    read_replay_file(args.file_name)


if __name__ == "__main__":
    main()
