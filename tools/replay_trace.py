#!/usr/bin/env python3
# ruff: noqa: E402
"""Text trace of one replay window: referee command and status message, ball, and the
nearest robot of each side, one line per referee change and every `--every` ticks.

The first thing to run on a stalled match or a voided restart, before rendering it:

    pixi run python tools/replay_trace.py replays/tournament_<id>/<a>_vs_<b>.npz 20 40

Times are seconds from the replay's first frame, the same clock as a `StallEvent`'s
`sim_time`. The status message (why play stopped, e.g. "Keep-out circle violation") comes
from the `.sparse_referee.pkl` beside the replay when there is one.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.entities.referee.referee_command import RefereeCommand


def _statuses(npz_path: Path, n: int) -> list[Optional[str]]:
    pkl = npz_path.with_name(npz_path.name[: -len(".npz")] + ".sparse_referee.pkl")
    out: list[Optional[str]] = [None] * n
    if pkl.exists():
        with open(pkl, "rb") as f:
            for i, data in pickle.load(f):
                if 0 <= i < n and data is not None:
                    out[i] = data.status_message
    return out


def _nearest(positions: np.ndarray, ids: np.ndarray, ball: np.ndarray) -> str:
    d = np.hypot(*(positions - ball).T)
    if not np.isfinite(d).any():
        return "-"
    k = int(np.nanargmin(d))
    return f"{int(ids[k])}@{d[k]:.2f}"


def trace(npz_path, t_start: float, t_end: float, every: int = 15) -> list[str]:
    npz_path = Path(npz_path)
    z = np.load(npz_path)
    ts = z["ts"] - z["ts"][0]
    statuses = _statuses(npz_path, len(ts))
    lines = []
    prev = None
    for i, t in enumerate(ts):
        cmd = RefereeCommand(int(z["referee_command"][i])).name if z["has_referee"][i] else "-"
        key = (cmd, statuses[i])
        if t_start <= t <= t_end and (key != prev or i % every == 0):
            ball = z["ball_p"][i, :2]
            speed = float(np.hypot(*z["ball_v"][i, :2]))
            holder = "F" if z["friendly_has_ball"][i].any() else "E" if z["enemy_has_ball"][i].any() else "-"
            lines.append(
                f"{t:7.2f} {cmd:<24} ball ({ball[0]:+.2f},{ball[1]:+.2f}) {speed:4.2f}m/s held={holder} "
                f"near F {_nearest(z['friendly_p'][i], z['friendly_ids'], ball):<8} "
                f"E {_nearest(z['enemy_p'][i], z['enemy_ids'], ball):<8}"
                + (f" | {statuses[i]}" if key != prev and statuses[i] else "")
            )
        prev = key
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, help="a tournament match's .npz")
    parser.add_argument("t_start", type=float)
    parser.add_argument("t_end", type=float)
    parser.add_argument("--every", type=int, default=15, help="also print every N ticks (default 15 = 0.25 s)")
    args = parser.parse_args()
    print("\n".join(trace(args.replay, args.t_start, args.t_end, args.every)))


if __name__ == "__main__":
    main()
