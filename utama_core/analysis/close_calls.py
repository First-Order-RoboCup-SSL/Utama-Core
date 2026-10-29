"""Close calls between robots in recorded field logs: where two robots' centres came within
`CLOSE_M` of each other, how near, how fast they were closing, and what the referee had on.

Reads SSL game logs written on real robots (`StrategyRunner(game_log_path=...)`): the camera's
raw detections (one camera per robot, `robot_calibration._main_camera`) and the referee's
commands. Each robot's track is the camera's view of its tag; pass a robot's tag offset
(`--tag`, as `tools/robot_calibration.py` takes it) to measure from the robot's centre.

Only the robots a run drove count (its run info's `vision_to_cmd`): ssl-vision misreads a tag
as another pattern now and then. A close call is one stretch with the two centres nearer than
`CLOSE_M`; contact is nearer than two robot radii, and nearer than one radius is one robot read
twice, left out. `approach_mps` is how fast the gap was shrinking at most in the `APPROACH_S`
before the stretch began: a slow nudge in a ball contest is not a head-on meeting. Tracks are
compared only where both robots were seen within `MAX_GAP_S`.

On the 2026-10-27/28 field logs: 35 close calls in 19 logs, 8 with contact, 13 closing at
0.9 m/s or more. 30 were in open play, 14 of them in `save_and_counter.py`'s 1v1 ball contests;
5 under a referee command (3 ball placements, a STOP, a HALT), all on 2026-10-27, before the
restart fixes of 75674c65. `examples/planner_ab.py` measures the planners alone.

    pixi run python -m utama_core.analysis.close_calls replays/field/2026102*_real_*.log
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from utama_core.analysis.robot_calibration import FieldLog, _main_camera
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.replay.ssl_log import REFEREE, read_records
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)

CLOSE_M = 0.25
CONTACT_M = 2 * ROBOT_RADIUS
# Nearer than this two "robots" are one robot read twice (a tag misread as another pattern, or
# one robot in both colours for a frame): real ones cannot overlap so far.
GHOST_M = ROBOT_RADIUS
APPROACH_S = 0.5
MAX_GAP_S = 0.2
STEP_S = 1 / 60
SMOOTH_S = 0.1  # the gap's change over this long: a one-frame jump in vision is not closing speed


def referee_commands(path) -> list[tuple[float, str]]:
    """(t, command name) at each change of the referee's command, t in seconds from the log's
    first record, as `FieldLog` times it."""
    out: list[tuple[float, str]] = []
    t0 = None
    msg = Referee()
    for ts_ns, msg_type, payload in read_records(path):
        t0 = ts_ns if t0 is None else t0
        if msg_type != REFEREE:
            continue
        msg.ParseFromString(payload)
        name = Referee.Command.Name(msg.command)
        if not out or out[-1][1] != name:
            out.append(((ts_ns - t0) / 1e9, name))
    return out


def _resampled(track: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y on `grid`, and whether a detection lies within `MAX_GAP_S` / 2 of each point."""
    t = track[:, 0]
    x, y = np.interp(grid, t, track[:, 2]), np.interp(grid, t, track[:, 3])
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    gap = np.minimum(np.abs(t[i] - grid), np.abs(t[i - 1] - grid))
    seen = (gap <= MAX_GAP_S / 2) & (grid >= t[0]) & (grid <= t[-1])
    return x, y, seen


def close_calls(
    robots: dict[tuple[str, int], np.ndarray], commands: list[tuple[float, str]] = (), close_m: float = CLOSE_M
) -> list[dict]:
    """Every close call between two of `robots` (`FieldLog.robots`: (t, camera, x, y, heading)
    rows per (colour, vision id)), with the referee command in force from `commands`."""
    tracks = {key: _main_camera(rows) for key, rows in robots.items() if len(rows) >= 2}
    keys = sorted(tracks)
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            start = max(tracks[a][0, 0], tracks[b][0, 0])
            end = min(tracks[a][-1, 0], tracks[b][-1, 0])
            if end - start < STEP_S:
                continue
            grid = np.arange(start, end, STEP_S)
            ax, ay, a_seen = _resampled(tracks[a], grid)
            bx, by, b_seen = _resampled(tracks[b], grid)
            distance = np.where(a_seen & b_seen, np.hypot(ax - bx, ay - by), np.inf)
            distance[distance < GHOST_M] = np.inf
            near = distance < close_m
            edges = np.flatnonzero(np.diff(np.r_[False, near, False].astype(int)))
            for s, e in zip(edges[::2], edges[1::2]):
                k = s + int(np.argmin(distance[s:e]))
                before = slice(max(0, s - int(APPROACH_S / STEP_S)), s + 1)
                d = distance[before]
                d = d[np.isfinite(d)]
                lag = int(round(SMOOTH_S / STEP_S))
                closing = (d[:-lag] - d[lag:]) / (lag * STEP_S) if len(d) > lag else np.array([])
                out.append(
                    {
                        "t": round(float(grid[s]), 2),
                        "robots": f"{a[0]} {a[1]} / {b[0]} {b[1]}",
                        "nearest_m": round(float(distance[k]), 3),
                        "seconds": round(float((e - s) * STEP_S), 2),
                        "approach_mps": round(float(closing.max()), 2) if len(closing) else None,
                        **_command_at(commands, float(grid[s])),
                    }
                )
    return sorted(out, key=lambda r: r["t"])


def _command_at(commands, t: float) -> dict:
    latest = None
    for since, name in commands:
        if since > t:
            break
        latest = (since, name)
    if latest is None:
        return {"command": None, "after_command_s": None}
    return {"command": latest[1], "after_command_s": round(t - latest[0], 2)}


def _tag_offsets(tags: list[list[str]]) -> dict[tuple[str, int], tuple[float, float, float]]:
    out = {}
    for robot, forward, left, heading in tags:
        colour, _, vision_id = robot.partition(":")
        out[(colour, int(vision_id))] = (float(forward), float(left), math.radians(float(heading)))
    return out


def scan(paths, tags: Optional[dict] = None, close_m: float = CLOSE_M) -> list[dict]:
    rows = []
    for path in paths:
        try:
            log = FieldLog(path)
        except ValueError as e:  # an empty log: the run stopped before vision came
            print(f"skipped: {e}")
            continue
        log.correct_tags(tags or {})
        driven = {
            (colour, int(vision_id))
            for colour, mapping in log.info.get("vision_to_cmd", {}).items()
            for vision_id in mapping
        }
        robots = {k: v for k, v in log.robots.items() if not driven or k in driven}
        rows += [{"log": Path(path).name, **r} for r in close_calls(robots, referee_commands(path), close_m)]
    return rows


def report(rows: list[dict]) -> str:
    lines = [f"{len(rows)} close calls, {sum(r['nearest_m'] < CONTACT_M for r in rows)} with contact"]
    by_command = Counter(r["command"] for r in rows)
    lines.append("  by referee command: " + ", ".join(f"{c} {n}" for c, n in by_command.most_common()))
    for r in rows:
        contact = " CONTACT" if r["nearest_m"] < CONTACT_M else ""
        approach = "-" if r["approach_mps"] is None else f"{r['approach_mps']:.2f} m/s"
        after = "" if r["after_command_s"] is None else f" +{r['after_command_s']:.1f} s"
        lines.append(
            f"  {r['log']}  t {r['t']:>7.1f} s  {r['robots']}  nearest {r['nearest_m']:.2f} m{contact}, "
            f"closing {approach}, {r['seconds']:.1f} s, {r['command']}{after}"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--close", type=float, default=CLOSE_M, help="metres, centre to centre (default %(default)s)")
    parser.add_argument(
        "--tag",
        nargs=4,
        action="append",
        default=[],
        metavar=("COLOUR:ID", "FORWARD_M", "LEFT_M", "HEADING_DEG"),
        help="where a robot's tag sits on it (RobotProfile.tag_*); repeatable",
    )
    parser.add_argument("--json", type=Path, help="also write the close calls here")
    args = parser.parse_args(argv)
    rows = scan(args.logs, _tag_offsets(args.tag), args.close)
    print(report(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
