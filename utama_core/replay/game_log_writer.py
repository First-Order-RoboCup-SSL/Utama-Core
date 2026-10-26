"""Record a run with a camera as an SSL game log: what the network and the radio carried.

Our replays hold the filtered world; measuring the real robots needs what went in and out
before any filter: every vision packet as received (raw detections with ssl-vision's
`t_capture`), the referee's state, and the radio's bytes both ways. `GameLogWriter` writes
them in the league's game-log format (`ssl_log.py`), so the tools for official logs read a
field test unchanged: `tools/convert_ssl_log.py` turns it into a replay, and its raw-vision
sidecar gives our camera's noise. Records:

- `VISION_2014`: each `SSL_WrapperPacket`, byte for byte.
- `REFEREE`: a `Referee` message whenever the referee's state changes. The in-process
  referee sends no packets, so the message is built from its `RefereeData`.
- `RADIO_OUT` / `RADIO_IN` (ours; other readers skip unknown types): each packet written to
  the radio (12 bytes per robot, `RealRobotController._generate_command_buffer`) and the
  feedback bytes read back. Commanded against seen motion gives the control delay.
- `RUN_INFO` (ours): one JSON object at the start: mode, colours, sides, each team's vision
  ID -> firmware ID mapping and robot profiles, so the radio's IDs can be matched to vision's.

Every record is stamped with the receive (or send) time on this machine's clock,
`time.time_ns()`, so vision and radio share one clock: their difference is the whole loop,
camera to robot to camera. The file is written uncompressed and flushed about once a second,
so a crash loses at most that second (`read_records` drops a cut-off last record).
"""

from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path
from typing import Optional

from utama_core.entities.data.referee import RefereeData
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)

# The league's format, as `ssl_log` reads it (not imported from there: the runner would pull
# in the whole reader).
LOG_HEADER = b"SSL_LOG_FILE"
REFEREE, VISION_2014 = 3, 4
RADIO_OUT, RADIO_IN, RUN_INFO = 100, 101, 102
_FLUSH_S = 1.0


def referee_message(data: RefereeData, command_counter: int) -> Referee:
    """`data` as the game controller would have sent it (times in us, positions in mm)."""
    msg = Referee(
        source_identifier=data.source_identifier or "utama",
        packet_timestamp=int(data.time_sent * 1e6),
        stage=data.stage.value,
        stage_time_left=int(data.stage_time_left * 1e6),
        command=data.referee_command.value,
        command_counter=command_counter,
        command_timestamp=int(data.referee_command_timestamp * 1e6),
    )
    for team, info in ((msg.yellow, data.yellow_team), (msg.blue, data.blue_team)):
        team.name = info.name
        team.score = info.score
        team.red_cards = info.red_cards
        team.yellow_card_times.extend(info.yellow_card_times)
        team.yellow_cards = info.yellow_cards
        team.timeouts = info.timeouts
        team.timeout_time = info.timeout_time
        team.goalkeeper = info.goalkeeper
    if data.designated_position is not None:
        msg.designated_position.x = data.designated_position[0] * 1000
        msg.designated_position.y = data.designated_position[1] * 1000
    if data.blue_team_on_positive_half is not None:
        msg.blue_team_on_positive_half = data.blue_team_on_positive_half
    if data.next_command is not None:
        msg.next_command = data.next_command.value
    return msg


class GameLogWriter:
    """Appends records to `path` (a `.log`); safe to call from the vision thread and the main loop."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "wb")
        self._file.write(LOG_HEADER + struct.pack(">i", 1))
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._referee_key: Optional[tuple] = None
        self._command_counter = 0

    def write(self, msg_type: int, payload: bytes, ts_ns: Optional[int] = None) -> None:
        ts_ns = time.time_ns() if ts_ns is None else ts_ns
        with self._lock:
            if self._file.closed:
                return
            self._file.write(struct.pack(">qii", ts_ns, msg_type, len(payload)) + payload)
            if time.monotonic() - self._last_flush > _FLUSH_S:
                self._file.flush()
                self._last_flush = time.monotonic()

    def run_info(self, info: dict) -> None:
        self.write(RUN_INFO, json.dumps(info).encode())

    def vision(self, packet: bytes) -> None:
        self.write(VISION_2014, packet)

    def radio_out(self, packet: bytes) -> None:
        self.write(RADIO_OUT, packet)

    def radio_in(self, data: bytes) -> None:
        self.write(RADIO_IN, data)

    def referee(self, data: RefereeData) -> None:
        """Records `data` when it differs from the last one recorded, clocks aside."""
        key = (
            data.referee_command,
            data.stage,
            data.yellow_team.score,
            data.blue_team.score,
            data.designated_position,
            data.next_command,
            data.blue_team_on_positive_half,
        )
        if key == self._referee_key:
            return
        if self._referee_key is None or key[0] != self._referee_key[0]:
            self._command_counter += 1
        self._referee_key = key
        self.write(REFEREE, referee_message(data, self._command_counter).SerializeToString())

    def close(self) -> None:
        with self._lock:
            self._file.close()
