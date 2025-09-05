import logging
import pickle
import warnings
from dataclasses import dataclass
from itertools import count
from typing import IO, Optional

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.entities.game import GameFrame
from utama_core.replay.entities import ReplayMetadata


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


class ReplayWriter:
    def __init__(
        self,
        replay_configs: ReplayWriterConfig,
        my_team_is_yellow: bool,
        exp_friendly: int,
        exp_enemy: int,
    ):
        self.logger = logging.getLogger(__name__)
        self.replay_configs = replay_configs
        self.file: Optional[IO] = self.create_file(
            replay_configs=replay_configs,
            replay_metadata=ReplayMetadata(
                my_team_is_yellow=my_team_is_yellow,
                exp_friendly=exp_friendly,
                exp_enemy=exp_enemy,
            ),
        )

    def create_file(self, replay_configs: ReplayWriterConfig, replay_metadata: ReplayMetadata):
        replay_path = REPLAY_BASE_PATH / f"{replay_configs.replay_name}.pkl"

        replay_path.parent.mkdir(parents=True, exist_ok=True)

        if replay_path.exists():
            if replay_configs.overwrite_existing:
                replay_path.write_bytes(b"")  # clear content

            else:
                for i in count(1):
                    candidate = REPLAY_BASE_PATH / f"{replay_configs.replay_name}_{i}.pkl"
                    if not candidate.exists():
                        replay_path = candidate
                        warnings.warn(f"Replay file already exists. Saving as {replay_path.name}")
                        break

        file = open(replay_path, "ab")
        try:
            pickle.dump(replay_metadata, file)
            file.flush()
        except Exception as e:
            self.logger.error(f"Failed to write replay metadata to file {replay_path}: {e}")
            file.close()
            return None
        return file

    def write_frame(self, frame: GameFrame):
        """Write a single game frame to the replay file."""
        if not self.file:
            self.logger.error("Replay file is not initialized.")
            return
        pickle.dump(frame, self.file)
        self.file.flush()

    def close(self):
        """Close the replay file."""
        if self.file:
            self.file.close()
            self.file = None
        else:
            self.logger.error("Replay file is not initialized.")
