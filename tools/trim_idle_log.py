"""Trim complete idle time sectors from SSL game logs (HALT/STOP dead time & trailing stretches).

Real-robot field logs often record long stretches of dead time (e.g. operators setting up
between drills or leaving the logger running after a session ends). This tool identifies
active segments (live play, restarts, or non-zero robot motion) and strips sustained idle
stretches longer than `--min-idle` seconds while retaining a safety buffer (`--pad`) around
transitions.

Usage:
    pixi run python tools/trim_idle_log.py replays/field/<run>.log.gz [--out OUT] [--dry-run]
    pixi run python tools/trim_idle_log.py replays/field/*.log.gz --in-place
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis.robot_calibration import decode_radio  # noqa: E402
from utama_core.replay.game_log_writer import (  # noqa: E402
    LOG_HEADER,
    NOTE,
    RADIO_IN,
    RADIO_OUT,
    REFEREE,
    RUN_INFO,
    VISION_2014,
)
from utama_core.replay.ssl_log import read_records  # noqa: E402
from utama_core.team_controller.generated_code.ssl_gc_referee_message_pb2 import (  # noqa: E402
    Referee,
)

# Referee commands representing active play or restarts
# 0 = HALT, 1 = STOP, 8/9 = TIMEOUT (idle)
# 2 = NORMAL_START, 3 = FORCE_START, 4/5 = PREPARE_KICKOFF, 6/7 = DIRECT_FREE, 10/11 = PREPARE_PENALTY
ACTIVE_COMMANDS = {2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17}


def is_active_radio(payload: bytes) -> bool:
    """True if any robot was commanded to translate, turn, dribble, or kick."""
    try:
        commands = decode_radio(payload)
        for _, fwd, left, turn, kicker, dribbler in commands:
            if abs(fwd) > 0.05 or abs(left) > 0.05 or abs(turn) > 0.1 or kicker or dribbler:
                return True
    except Exception:
        pass
    return False


def find_active_intervals(
    records: list[tuple[int, int, bytes]],
    pad_s: float = 1.0,
    min_idle_s: float = 5.0,
) -> list[tuple[int, int]]:
    """Return a list of (start_ns, end_ns) time windows to keep."""
    if not records:
        return []

    t0_ns = records[0][0]
    t_end_ns = records[-1][0]
    pad_ns = int(pad_s * 1e9)
    min_idle_ns = int(min_idle_s * 1e9)

    intervals: list[tuple[int, int]] = []
    active_start: Optional[int] = None
    last_active_ts: Optional[int] = None
    current_ref_active = False

    for ts_ns, msg_type, payload in records:
        is_active = False
        if msg_type == REFEREE:
            try:
                ref = Referee()
                ref.ParseFromString(payload)
                new_active = ref.command in ACTIVE_COMMANDS
            except Exception:
                new_active = False
            if current_ref_active and not new_active:
                last_active_ts = ts_ns
            current_ref_active = new_active
            is_active = current_ref_active
        elif current_ref_active:
            is_active = True
        elif msg_type == RADIO_OUT and is_active_radio(payload):
            is_active = True

        if is_active:
            if active_start is None:
                active_start = ts_ns
            last_active_ts = ts_ns
        else:
            if active_start is not None and last_active_ts is not None:
                intervals.append((max(t0_ns, active_start - pad_ns), min(t_end_ns, last_active_ts + pad_ns)))
                active_start = None
                last_active_ts = None

    if active_start is not None and last_active_ts is not None:
        intervals.append((max(t0_ns, active_start - pad_ns), min(t_end_ns, last_active_ts + pad_ns)))

    if not intervals:
        return [(t0_ns, t_end_ns)]

    # Merge overlapping or close windows
    merged: list[tuple[int, int]] = []
    curr_start, curr_end = intervals[0]
    for w_start, w_end in intervals[1:]:
        if w_start <= curr_end + min_idle_ns:
            curr_end = max(curr_end, w_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = w_start, w_end
    merged.append((curr_start, curr_end))
    return merged


def trim_log(
    src_path: Path,
    out_path: Optional[Path] = None,
    pad_s: float = 1.0,
    min_idle_s: float = 5.0,
    dry_run: bool = False,
) -> dict:
    """Trim dead/idle time sectors from a single game log."""
    src_path = Path(src_path)
    records = list(read_records(src_path))
    if not records:
        return {"file": src_path.name, "status": "empty"}

    orig_records_count = len(records)
    orig_duration_s = (records[-1][0] - records[0][0]) / 1e9
    orig_bytes = src_path.stat().st_size

    intervals = find_active_intervals(records, pad_s=pad_s, min_idle_s=min_idle_s)

    # Determine which records to keep
    # Always keep initial RUN_INFO, NOTE, and the first REFEREE message
    keep_records: list[tuple[int, int, bytes]] = []
    first_referee_kept = False

    for r in records:
        ts_ns, msg_type, _ = r
        if msg_type in (RUN_INFO, NOTE):
            keep_records.append(r)
            continue
        if msg_type == REFEREE and not first_referee_kept:
            keep_records.append(r)
            first_referee_kept = True
            continue

        # Check if inside any kept interval
        in_interval = False
        for start_ns, end_ns in intervals:
            if start_ns <= ts_ns <= end_ns:
                in_interval = True
                break
            if ts_ns < start_ns:
                break
        if in_interval:
            keep_records.append(r)

    kept_duration_s = sum((end - start) / 1e9 for start, end in intervals) if intervals else 0.0

    stats = {
        "file": src_path.name,
        "orig_duration_s": orig_duration_s,
        "kept_duration_s": kept_duration_s,
        "duration_reduction_pct": (1.0 - kept_duration_s / max(orig_duration_s, 0.001)) * 100,
        "orig_records": orig_records_count,
        "kept_records": len(keep_records),
        "orig_bytes": orig_bytes,
    }

    if dry_run:
        return stats

    # Write output
    target = out_path or src_path
    tmp_out = target.with_name(f"{target.name}.trimming.tmp")

    opener = gzip.open if target.suffix == ".gz" else open
    with opener(tmp_out, "wb") as f:
        f.write(LOG_HEADER + struct.pack(">i", 1))
        for ts_ns, msg_type, payload in keep_records:
            head = struct.pack(">qii", ts_ns, msg_type, len(payload))
            f.write(head + payload)

    tmp_out.replace(target)
    stats["new_bytes"] = target.stat().st_size
    stats["bytes_saved_pct"] = (1.0 - stats["new_bytes"] / max(orig_bytes, 1)) * 100
    return stats


def _process_one(args_tuple: tuple[Path, Optional[Path], float, float, bool, bool]) -> dict:
    path, out, pad, min_idle, in_place, dry_run = args_tuple
    out_dest = out if out else (path if in_place else path.with_name(f"{path.stem}.trimmed{path.suffix}"))
    return trim_log(
        path,
        out_path=out_dest,
        pad_s=pad,
        min_idle_s=min_idle,
        dry_run=dry_run,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Trim idle sectors and trailing dead time from SSL game logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Log file(s) or glob to trim.")
    parser.add_argument("--out", type=Path, default=None, help="Output file (single log only).")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input log(s) with trimmed version.")
    parser.add_argument(
        "--pad", type=float, default=1.0, help="Seconds of buffer around active windows (default: 1.0)."
    )
    parser.add_argument(
        "--min-idle", type=float, default=5.0, help="Minimum idle seconds to trigger cutting (default: 5.0)."
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers (default: 8).")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files.")

    args = parser.parse_args(argv)

    if args.out and len(args.logs) > 1:
        parser.error("--out can only be specified for a single input log file.")

    valid_paths = [p for p in args.logs if p.is_file()]

    tasks = [(p, args.out, args.pad, args.min_idle, args.in_place, args.dry_run) for p in valid_paths]

    if len(tasks) <= 1 or args.workers == 1:
        results = [_process_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            results = list(executor.map(_process_one, tasks))

    total_orig_bytes = sum(s.get("orig_bytes", 0) for s in results)
    total_new_bytes = sum(s.get("new_bytes", s.get("orig_bytes", 0)) for s in results)
    total_orig_s = sum(s.get("orig_duration_s", 0) for s in results)
    total_kept_s = sum(s.get("kept_duration_s", 0) for s in results)

    for stats in results:
        print(
            f"{stats['file']}: {stats.get('orig_duration_s', 0):.1f}s -> {stats.get('kept_duration_s', 0):.1f}s "
            f"({stats.get('duration_reduction_pct', 0):.1f}% cut, "
            f"{stats.get('orig_records', 0)} -> {stats.get('kept_records', 0)} records)"
        )

    if len(results) > 1:
        saved_bytes = total_orig_bytes - total_new_bytes
        print("\n--- Summary ---")
        print(f"Processed {len(results)} logs:")
        print(
            f"Total time: {total_orig_s:.1f}s -> {total_kept_s:.1f}s ({(1.0 - total_kept_s/max(total_orig_s,1))*100:.1f}% cut)"
        )
        if not args.dry_run:
            print(
                f"Storage: {total_orig_bytes / (1024*1024):.1f} MB -> {total_new_bytes / (1024*1024):.1f} MB (saved {saved_bytes / (1024*1024):.1f} MB, {saved_bytes / max(total_orig_bytes,1)*100:.1f}%)"
            )


if __name__ == "__main__":
    main()
