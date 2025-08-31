import logging
import pickle
import warnings
from dataclasses import dataclass
from itertools import count

from config.settings import REPLAY_BASE_PATH
from entities.game.game import Game


@dataclass(kw_only=True)
class ReplayWriterConfig:
    """Configuration settings for initializing a ReplayWriter.

    Attributes:
        replay_name (str): The name of the replay file or session to be written.
        is_my_perspective (bool, optional): Whether to record the replay from
            the user's perspective or opponent's. Defaults to True.
        overwrite_existing (bool, optional): Whether to overwrite existing replay with same name. Defaults to False.
    """

    replay_name: str
    is_my_perspective: bool = True
    overwrite_existing: bool = False


def write_replay(replay_configs: ReplayWriterConfig, game_to_write: Game):
    replay_path = REPLAY_BASE_PATH / f"{replay_configs.replay_name}.pkl"

    if not replay_configs.overwrite_existing and replay_path.exists():
        for i in count(1):
            candidate = REPLAY_BASE_PATH / f"{replay_configs.replay_name}_{i}.pkl"
            if not candidate.exists():
                replay_path = candidate
                warnings.warn(f"Replay file already exists. Saving as {replay_path.name}")
                break

    with open(replay_path, "wb") as f:
        pickle.dump(game_to_write, f)
