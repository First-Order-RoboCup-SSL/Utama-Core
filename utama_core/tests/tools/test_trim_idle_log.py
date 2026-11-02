"""Tests for tools/trim_idle_log.py: trimming dead/idle time sectors from SSL logs."""

import gzip
import struct
from pathlib import Path

from tools.trim_idle_log import find_active_intervals, trim_log
from utama_core.replay.game_log_writer import LOG_HEADER, REFEREE, RUN_INFO
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)


def _write_synthetic_records(path: Path, records: list[tuple[int, int, bytes]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wb") as f:
        f.write(LOG_HEADER + struct.pack(">i", 1))
        for ts_ns, msg_type, payload in records:
            head = struct.pack(">qii", ts_ns, msg_type, len(payload))
            f.write(head + payload)


def _ref_msg(cmd: int) -> bytes:
    ref = Referee()
    ref.packet_timestamp = 0
    ref.stage = 0
    ref.command_counter = 1
    ref.command_timestamp = 0
    for team, name in ((ref.yellow, "Yellow"), (ref.blue, "Blue")):
        team.name = name
        team.score = 0
        team.red_cards = 0
        team.yellow_cards = 0
        team.timeouts = 0
        team.timeout_time = 0
        team.goalkeeper = 0
    ref.command = cmd
    return ref.SerializeToString()


def test_find_active_intervals_trims_sustained_idle_and_trailing_halt():
    # 0 = HALT, 3 = FORCE_START
    records = [
        (0, REFEREE, _ref_msg(0)),  # t = 0s: HALT
        (10_000_000_000, REFEREE, _ref_msg(3)),  # t = 10s: FORCE_START (active)
        (20_000_000_000, REFEREE, _ref_msg(0)),  # t = 20s: HALT
        (50_000_000_000, REFEREE, _ref_msg(0)),  # t = 50s: still HALT (30s trailing dead time)
    ]

    intervals = find_active_intervals(records, pad_s=1.0, min_idle_s=5.0)

    # Active window should span around t = 10s to 20s (+/- pad)
    assert len(intervals) == 1
    start_ns, end_ns = intervals[0]
    assert 9_000_000_000 <= start_ns <= 10_000_000_000
    assert 20_000_000_000 <= end_ns <= 21_000_000_000


def test_trim_log_preserves_run_info_and_reduces_size(tmp_path):
    log_path = tmp_path / "drill.log.gz"
    records = [
        (0, RUN_INFO, b'{"mode": "real"}'),
        (1_000_000_000, REFEREE, _ref_msg(0)),  # t = 1s: HALT
        (10_000_000_000, REFEREE, _ref_msg(3)),  # t = 10s: FORCE_START
        (15_000_000_000, REFEREE, _ref_msg(3)),  # t = 15s: FORCE_START
        (16_000_000_000, REFEREE, _ref_msg(0)),  # t = 16s: HALT
        (60_000_000_000, REFEREE, _ref_msg(0)),  # t = 60s: HALT (trailing 44s dead time)
    ]
    _write_synthetic_records(log_path, records)

    out_path = tmp_path / "drill.trimmed.log.gz"
    stats = trim_log(log_path, out_path=out_path, pad_s=1.0, min_idle_s=5.0)

    assert stats["duration_reduction_pct"] > 60.0
    assert stats["kept_records"] < stats["orig_records"]
    assert out_path.exists()

    # Verify RUN_INFO was preserved
    from utama_core.replay.ssl_log import read_records

    trimmed_records = list(read_records(out_path))
    msg_types = [m for _, m, _ in trimmed_records]
    assert RUN_INFO in msg_types
