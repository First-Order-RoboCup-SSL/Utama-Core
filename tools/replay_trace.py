#!/usr/bin/env python3
# ruff: noqa: E402
"""Text trace of one replay window: referee command and status message, ball, and the
nearest robot of each side, one line per referee change and every `--every` ticks.

The first thing to run on a stalled match or a voided restart, before rendering it:

    pixi run python tools/replay_trace.py replays/tournament_<id>/<a>_vs_<b>.npz 20 40

`--events` prints the events in the window instead of the periodic rows, one line each:
  KICK   the ball's speed rises by >= 0.5 m/s in one tick, or leaves rest (< 0.5 to >= 1.0 m/s);
         counted once per 0.25 s. The kicker is the robot nearest the ball at that tick.
  HOLD   possession changes (F = friendly, E = enemy, - = nobody), as the `held=` column.
  REF    the referee command changes, with the status message when there is one.
  FOUL   a foul from the replay's `.stats.json` (ours = friendly, theirs = enemy).
At most 40 lines; the rest are counted as "... N more".

`--json` writes the rows (or the events, with `--events`) as a JSON list instead of text.
Each row: t, referee (command name), status, ball [x, y], ball_speed, held (F/E/-), near_F and
near_E ({id, dist} or null). Each event: t, kind (KICK/HOLD/REF/FOUL), text, and kind-specific
fields (KICK: side, robot_id, dist, ball_speed; HOLD: from, to; REF: command, status; FOUL:
side, rule, robot_id, tactic).

Times are seconds from the replay's first frame, the same clock as a `StallEvent`'s
`sim_time`. The status message (why play stopped, e.g. "Keep-out circle violation") comes
from the `.sparse_referee.pkl` beside the replay when there is one.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # before any import: hides pygame's banner
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.entities.referee.referee_command import RefereeCommand

MAX_EVENT_LINES = 40
KICK_RISE_MS = 0.5  # m/s rise in one tick
KICK_FROM_REST_MS = 0.5  # below this, then >= KICK_TO_MS, counts as a kick
KICK_TO_MS = 1.0
KICK_COOLDOWN_S = 0.25


def _statuses(npz_path: Path, n: int) -> list[Optional[str]]:
    pkl = npz_path.with_name(npz_path.name[: -len(".npz")] + ".sparse_referee.pkl")
    out: list[Optional[str]] = [None] * n
    if pkl.exists():
        with open(pkl, "rb") as f:
            for i, data in pickle.load(f):
                if 0 <= i < n and data is not None:
                    out[i] = data.status_message
    return out


def _fouls(npz_path: Path) -> list[dict]:
    """Fouls from the `.stats.json` beside the replay (`sim_time`, `side`: friendly or enemy)."""
    stats = npz_path.with_name(npz_path.name[: -len(".npz")] + ".stats.json")
    if not stats.exists():
        return []
    return json.loads(stats.read_text()).get("fouls") or []


def _nearest(positions: np.ndarray, ids: np.ndarray, ball: np.ndarray) -> Optional[tuple[int, float]]:
    """(robot id, distance) of the robot nearest `ball`, or None if no robot has a position."""
    d = np.hypot(*(positions - ball).T)
    if not np.isfinite(d).any():
        return None
    k = int(np.nanargmin(d))
    return int(ids[k]), float(d[k])


def _near_text(near: Optional[dict]) -> str:
    return "-" if near is None else f"{near['id']}@{near['dist']:.2f}"


def _state(z, statuses: list[Optional[str]], i: int) -> dict:
    """The row for tick `i` of a replay (`z` is the npz, or a dict with the same arrays)."""
    cmd = RefereeCommand(int(z["referee_command"][i])).name if z["has_referee"][i] else "-"
    ball = z["ball_p"][i, :2]
    speed = float(np.hypot(*z["ball_v"][i, :2]))
    holder = "F" if z["friendly_has_ball"][i].any() else "E" if z["enemy_has_ball"][i].any() else "-"
    near = {}
    for side in ("friendly", "enemy"):
        n = _nearest(z[f"{side}_p"][i], z[f"{side}_ids"], ball)
        near[side] = None if n is None else {"id": n[0], "dist": n[1]}
    return {
        "t": float(z["ts"][i] - z["ts"][0]),
        "referee": cmd,
        "status": statuses[i],
        "ball": [float(ball[0]), float(ball[1])],
        "ball_speed": speed,
        "held": holder,
        "near_F": near["friendly"],
        "near_E": near["enemy"],
    }


def _row_text(r: dict, status_changed: bool) -> str:
    x, y = r["ball"]
    return (
        f"{r['t']:7.2f} {r['referee']:<24} ball ({x:+.2f},{y:+.2f}) {r['ball_speed']:4.2f}m/s held={r['held']} "
        f"near F {_near_text(r['near_F']):<8} E {_near_text(r['near_E']):<8}"
        + (f" | {r['status']}" if status_changed and r["status"] else "")
    )


def rows(z, statuses: list[Optional[str]], t_start: float, t_end: float, every: int = 15) -> list[dict]:
    """The periodic rows: one at each referee or status change and every `every` ticks, in the window."""
    out = []
    prev = None
    ts = z["ts"] - z["ts"][0]
    for i, t in enumerate(ts):
        key = (
            RefereeCommand(int(z["referee_command"][i])).name if z["has_referee"][i] else "-",
            statuses[i],
        )
        if t_start <= t <= t_end and (key != prev or i % every == 0):
            r = _state(z, statuses, i)
            r["status_changed"] = key != prev
            out.append(r)
        prev = key
    return out


def events(z, statuses: list[Optional[str]], fouls: list[dict], t_start: float, t_end: float) -> list[dict]:
    """The KICK, HOLD, REF and FOUL events with t in [t_start, t_end], in time order."""
    ts = z["ts"] - z["ts"][0]
    speed = np.hypot(z["ball_v"][:, 0], z["ball_v"][:, 1])
    out: list[dict] = []
    first = max(1, int(np.searchsorted(ts, t_start, side="left")))
    if first < len(ts):
        prev = _state(z, statuses, first - 1)
        last_kick = -np.inf
        for i in range(first, len(ts)):
            t = float(ts[i])
            if t > t_end:
                break
            cur = _state(z, statuses, i)
            rise = speed[i] - speed[i - 1]
            is_kick = (speed[i - 1] < KICK_FROM_REST_MS and speed[i] >= KICK_TO_MS) or rise >= KICK_RISE_MS
            if is_kick and t - last_kick >= KICK_COOLDOWN_S:
                last_kick = t
                kick = {"side": None, "robot_id": None, "dist": None}
                near = [(s, cur[k]) for s, k in (("F", "near_F"), ("E", "near_E")) if cur[k] is not None]
                if near:
                    side, n = min(near, key=lambda sn: sn[1]["dist"])
                    kick = {"side": "ours" if side == "F" else "theirs", "robot_id": n["id"], "dist": n["dist"]}
                    who = f"{side}{n['id']} at {n['dist']:.2f} m"
                else:
                    who = "no robot"
                out.append(
                    {
                        "t": t,
                        "kind": "KICK",
                        **kick,
                        "ball_speed": float(speed[i]),
                        "text": f"{t:7.2f} KICK  by {who}, ball {speed[i - 1]:.2f} -> {speed[i]:.2f} m/s",
                    }
                )
            if cur["held"] != prev["held"]:
                out.append(
                    {
                        "t": t,
                        "kind": "HOLD",
                        "from": prev["held"],
                        "to": cur["held"],
                        "text": f"{t:7.2f} HOLD  {prev['held']} -> {cur['held']}",
                    }
                )
            if cur["referee"] != prev["referee"] or (cur["status"] != prev["status"] and cur["status"]):
                status = f" | {cur['status']}" if cur["status"] else ""
                out.append(
                    {
                        "t": t,
                        "kind": "REF",
                        "command": cur["referee"],
                        "status": cur["status"],
                        "text": f"{t:7.2f} REF   {prev['referee']} -> {cur['referee']}{status}",
                    }
                )
            prev = cur
    for f in fouls:
        ft = float(f["sim_time"])
        if t_start <= ft <= t_end:
            side = "ours" if f.get("side") == "friendly" else "theirs"
            out.append(
                {
                    "t": ft,
                    "kind": "FOUL",
                    "side": side,
                    "rule": f["rule"],
                    "robot_id": f.get("robot_id"),
                    "tactic": f.get("tactic"),
                    "text": f"{ft:7.2f} FOUL  {side} {f['rule']} robot {f.get('robot_id')} {f.get('tactic') or ''}".rstrip(),
                }
            )
    out.sort(key=lambda e: e["t"])
    return out


def _load(npz_path: Path):
    z = dict(np.load(npz_path))  # NpzFile re-reads an array on every access; load each once
    return z, _statuses(npz_path, len(z["ts"])), _fouls(npz_path)


def trace(npz_path, t_start: float, t_end: float, every: int = 15) -> list[str]:
    """The periodic rows of the window as text lines."""
    z, statuses, _ = _load(Path(npz_path))
    return [_row_text(r, r["status_changed"]) for r in rows(z, statuses, t_start, t_end, every)]


def event_lines(npz_path, t_start: float, t_end: float) -> list[str]:
    """The events of the window as text lines, capped at MAX_EVENT_LINES."""
    z, statuses, fouls = _load(Path(npz_path))
    evs = events(z, statuses, fouls, t_start, t_end)
    lines = [e["text"] for e in evs[:MAX_EVENT_LINES]]
    if len(evs) > MAX_EVENT_LINES:
        lines.append(f"... {len(evs) - MAX_EVENT_LINES} more")
    return lines


def _jsonable(x):
    """Floats to 3 decimals and event texts stripped, so the JSON is short and stable."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, str):
        return x.strip()
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, help="a tournament match's .npz")
    parser.add_argument("t_start", type=float)
    parser.add_argument("t_end", type=float)
    parser.add_argument("--every", type=int, default=15, help="also print every N ticks (default 15 = 0.25 s)")
    parser.add_argument("--events", action="store_true", help="print the events (kicks, possession, referee, fouls)")
    parser.add_argument("--json", action="store_true", help="write JSON (rows, or events with --events)")
    args = parser.parse_args()
    if args.json:
        z, statuses, fouls = _load(args.replay)
        if args.events:
            data = events(z, statuses, fouls, args.t_start, args.t_end)
        else:
            data = [
                {k: v for k, v in r.items() if k != "status_changed"}
                for r in rows(z, statuses, args.t_start, args.t_end, args.every)
            ]
        print(json.dumps(_jsonable(data), indent=1))
    elif args.events:
        print("\n".join(event_lines(args.replay, args.t_start, args.t_end)))
    else:
        print("\n".join(trace(args.replay, args.t_start, args.t_end, args.every)))


if __name__ == "__main__":
    main()
