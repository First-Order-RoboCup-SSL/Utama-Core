"""What became of every restart in a tournament run: taken, or why not.

`round_robin.py` runs this after every saved run and writes `summary()` into
`summary.json` under `restarts`. Found necessary 2026-09-28: 8 of 19 penalties in a full
round-robin were never taken (the ball was never placed, then keep-out voided them) and no
existing signal showed it.

A restart episode starts at a restart command (`PREPARE_KICKOFF_*`, `DIRECT_FREE_*`,
`PREPARE_PENALTY_*`) and ends as one of:

- `taken` — NORMAL_START, then the ball moved `TAKEN_BALL_MOVE_M` from where it stood;
- `voided` — left for anything but NORMAL_START (e.g. STOP after a keep-out foul);
- `stopped_before_kick` — NORMAL_START, then another command before the ball moved;
- `timeout` — NORMAL_START, then FORCE_START before the ball moved;
- `match_ended` — still open when the replay ends.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from utama_core.entities.referee.referee_command import RefereeCommand as C

_RESTART_COMMANDS = {
    C.PREPARE_KICKOFF_YELLOW,
    C.PREPARE_KICKOFF_BLUE,
    C.DIRECT_FREE_YELLOW,
    C.DIRECT_FREE_BLUE,
    C.PREPARE_PENALTY_YELLOW,
    C.PREPARE_PENALTY_BLUE,
}
TAKEN_BALL_MOVE_M = 0.05

Row = tuple[float, Optional[C], Optional[tuple[float, float]]]


def episodes(rows: Iterable[Row]) -> list[dict]:
    """Restart episodes over per-tick `(t, command, ball_xy)` rows, in order."""
    out: list[dict] = []
    cur: Optional[dict] = None
    ball0: Optional[tuple[float, float]] = None
    prev = None
    for t, cmd, ball in rows:
        if cmd != prev:
            if cur is not None and cur["outcome"] is None and cur["reached_normal_start"]:
                cur["outcome"] = "timeout" if cmd == C.FORCE_START else "stopped_before_kick"
            if cmd in _RESTART_COMMANDS:
                if cur is None or cur["outcome"] is not None:
                    cur = {"kind": cmd.name.rsplit("_", 1)[0], "t": t, "reached_normal_start": False, "outcome": None}
                    out.append(cur)
            elif cur is not None and cur["outcome"] is None:
                if cmd == C.NORMAL_START:
                    cur["reached_normal_start"] = True
                    ball0 = ball
                else:
                    cur["outcome"] = "voided"
            prev = cmd
        if (
            cur is not None
            and cur["outcome"] is None
            and cur["reached_normal_start"]
            and ball is not None
            and ball0 is not None
            and math.hypot(ball[0] - ball0[0], ball[1] - ball0[1]) >= TAKEN_BALL_MOVE_M
        ):
            cur["outcome"] = "taken"
    for e in out:
        if e["outcome"] is None:
            e["outcome"] = "match_ended"
    return out


def analyse_match(npz_path) -> list[dict]:
    """`episodes` of one columnar replay, `t` in seconds from its first frame."""
    path = Path(npz_path)
    z = np.load(path)
    ts = z["ts"] - z["ts"][0]
    commands = [C(int(c)) if has else None for c, has in zip(z["referee_command"], z["has_referee"])]
    balls = [None if np.isnan(b[0]) else (float(b[0]), float(b[1])) for b in z["ball_p"]]
    match = path.name[: -len(".npz")]
    return [{"match": match, **e} for e in episodes(zip(ts.tolist(), commands, balls))]


def summarise(eps: list[dict]) -> dict:
    """Counts for `summary.json`: overall, then per restart kind with outcomes."""
    by_kind: dict[str, dict] = {}
    for e in eps:
        row = by_kind.setdefault(e["kind"], {"n": 0, "reached_normal_start": 0, "outcomes": collections.Counter()})
        row["n"] += 1
        row["reached_normal_start"] += e["reached_normal_start"]
        row["outcomes"][e["outcome"]] += 1
    for row in by_kind.values():
        row["outcomes"] = dict(row["outcomes"].most_common())
    return {
        "restarts": len(eps),
        "reached_normal_start": sum(e["reached_normal_start"] for e in eps),
        "by_kind": dict(sorted(by_kind.items())),
    }
