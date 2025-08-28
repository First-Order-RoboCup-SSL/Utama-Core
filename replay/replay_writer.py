import logging
from dataclasses import dataclass
import pickle
from typing import IO, Optional


@dataclass(kw_only=True)
class ReplayWriterConfig:
    """
    Configuration settings for initializing a ReplayWriter.

    Attributes:
        replay_name (str): The name of the replay file or session to be written.
        is_my_perspective (bool, optional): Whether to record the replay from
            the user's perspective or opponent's. Defaults to True.
    """

    replay_name: str
    is_my_perspective: bool = True


class ReplayWriter:
    def __init__(self, replay_writer_config: ReplayWriterConfig):
        self.replay_name = replay_writer_config.replay_name
        self.is_my_perspective = replay_writer_config.is_my_perspective
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Replay recording enabled. Saving to {replay_name}")
        self.replay_file: Optional[IO] = None

    def write_frame(self):
        pass``

    def close(self):
        if self.replay_file:
            self.replay_file.close()
        self.logger.info(f"Replay saved to {self.replay_name}")
