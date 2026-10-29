"""Where a robot's vision tag sits on it, from a recorded field test: its offset from the robot's
centre and how far its heading is off the robot's forward direction.

Reads an SSL game log written on real robots (`StrategyRunner(game_log_path=...)`): the turns
and straight drives of an `examples/calibrate_robot.py` run, which writes the offset into the
robot's profile.

- **Tag offset**: a robot turning on the spot turns about its own centre, so its tag traces a
  circle round that centre. Per turn on the spot held at least `MIN_SPIN_S`, after `SETTLE_S`, a
  least-squares fit of `tag = centre + drift * t + R(heading) @ offset` gives `offset`: (forward,
  left) of the tag from the centre in the frame vision reports the robot in, metres. The robot's
  centre is `tag - R(heading) @ offset`.
- **Drive direction**: per straight drive (forward or backward only) held at least `MIN_DRIVE_S`,
  after `SETTLE_S`, the direction the tag travelled against its reported heading (degrees,
  anticlockwise +). It is the tag's heading offset plus any sideways pull of the wheels; the two
  cannot be told apart by driving. A tag's heading offset needs no measuring of its own: the
  kicks' bias is fitted against the tag's heading (`RobotProfile.kick_bias_deg`).

    pixi run python -m utama_core.analysis.tag_calibration replays/field/<run>.log
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from utama_core.analysis.robot_calibration import FieldLog, _main_camera

MIN_SPIN_S = 1.5
MIN_DRIVE_S = 0.8
SETTLE_S = 0.4  # the start of a move left out: loop delay and acceleration
MIN_DRIVE_M = 0.15  # a drive shorter than this says little about its direction


def _held(commands: np.ndarray, keep, min_s: float) -> list[tuple[float, float, np.ndarray]]:
    """(start, end, command) of every constant (forward, left, turn) command `keep` accepts,
    held at least `min_s`."""
    out = []
    i = 0
    n = len(commands)
    while i < n:
        j = i
        while j + 1 < n and np.array_equal(commands[j + 1, 1:4], commands[i, 1:4]):
            j += 1
        end = commands[j + 1, 0] if j + 1 < n else commands[j, 0]
        if keep(commands[i, 1:4]) and end - commands[i, 0] >= min_s:
            out.append((float(commands[i, 0]), float(end), commands[i, 1:4]))
        i = j + 1
    return out


def fit_offset(t: np.ndarray, x: np.ndarray, y: np.ndarray, heading: np.ndarray) -> tuple[float, float, float]:
    """(forward, left, rms residual) of the tag from the centre it turns about; see the module."""
    c, s = np.cos(heading), np.sin(heading)
    zero, one = np.zeros_like(t), np.ones_like(t)
    rows_x = np.column_stack([one, zero, t - t[0], zero, c, -s])
    rows_y = np.column_stack([zero, one, zero, t - t[0], s, c])
    a = np.vstack([rows_x, rows_y])
    b = np.concatenate([x, y])
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    rms = float(np.sqrt(np.mean((a @ sol - b) ** 2)))
    return float(sol[4]), float(sol[5]), rms


def _window(track: np.ndarray, start: float, end: float) -> np.ndarray:
    return track[(track[:, 0] >= start + SETTLE_S) & (track[:, 0] <= end)]


def spins(commands: np.ndarray, track: np.ndarray) -> list[dict]:
    """The tag offset fitted to each turn on the spot."""
    out = []
    for start, end, cmd in _held(commands, lambda c: c[2] != 0 and c[0] == 0 and c[1] == 0, MIN_SPIN_S):
        w = _window(track, start, end)
        heading = np.unwrap(w[:, 4]) if len(w) else w[:, 4]
        if len(w) < 20 or abs(heading[-1] - heading[0]) < math.pi:
            continue  # less than half a turn: the circle is not pinned down
        fwd, left, rms = fit_offset(w[:, 0], w[:, 2], w[:, 3], heading)
        out.append(
            {
                "t": round(start, 2),
                "turn": float(cmd[2]),
                "turned_deg": round(math.degrees(abs(heading[-1] - heading[0]))),
                "forward_m": round(fwd, 4),
                "left_m": round(left, 4),
                "rms_m": round(rms, 4),
            }
        )
    return out


def drives(commands: np.ndarray, track: np.ndarray) -> list[dict]:
    """The tag's direction of travel against its heading on each straight drive."""
    out = []
    for start, end, cmd in _held(commands, lambda c: c[0] != 0 and c[1] == 0 and c[2] == 0, MIN_DRIVE_S):
        w = _window(track, start, end)
        if len(w) < 10:
            continue
        dx, dy = w[-1, 2] - w[0, 2], w[-1, 3] - w[0, 3]
        if math.hypot(dx, dy) < MIN_DRIVE_M:
            continue
        travel = math.atan2(dy, dx) + (math.pi if cmd[0] < 0 else 0.0)
        heading = math.atan2(np.mean(np.sin(w[:, 4])), np.mean(np.cos(w[:, 4])))
        off = math.remainder(travel - heading, math.tau)
        out.append({"t": round(start, 2), "forward": float(cmd[0]), "direction_deg": round(math.degrees(off), 2)})
    return out


def _median(values: list[float]) -> Optional[float]:
    return round(float(np.median(values)), 4) if values else None


def tag_calibration(log: FieldLog) -> list[dict]:
    """One result per robot sent any motion."""
    results = []
    for fw, commands in sorted(log.commands.items()):
        robot = log.robot_of(fw)
        if robot is None or robot not in log.robots or not np.any(commands[:, 1:4] != 0):
            continue
        track = _main_camera(log.robots[robot])
        spin_rows, drive_rows = spins(commands, track), drives(commands, track)
        results.append(
            {
                "robot": f"{robot[0]} {robot[1]}",
                "firmware_id": fw,
                "tag_forward_m": _median([r["forward_m"] for r in spin_rows]),
                "tag_left_m": _median([r["left_m"] for r in spin_rows]),
                "drive_direction_deg": _median([r["direction_deg"] for r in drive_rows]),
                "spins": spin_rows,
                "drives": drive_rows,
            }
        )
    return results


def report(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"{r['robot']} (radio {r['firmware_id']})")
        if r["tag_forward_m"] is None:
            lines.append("  tag offset        no turn on the spot of half a turn or more")
        else:
            fwd, left = r["tag_forward_m"] * 100, r["tag_left_m"] * 100
            lines.append(f"  tag offset        {fwd:+.1f} cm forward, {left:+.1f} cm left of the centre (median)")
            for s in r["spins"]:
                lines.append(
                    f"    turn {s['turn']:+.1f} rad/s at {s['t']:.0f} s, {s['turned_deg']}°: "
                    f"{s['forward_m'] * 100:+.1f} cm, {s['left_m'] * 100:+.1f} cm (fit rms {s['rms_m'] * 100:.1f} cm)"
                )
        if r["drive_direction_deg"] is None:
            lines.append("  drive direction   no straight drive")
        else:
            lines.append(
                f"  drive direction   {r['drive_direction_deg']:+.1f}° off the tag's heading (median, anticlockwise +)"
            )
            lines.append(
                "    " + ", ".join(f"{d['forward']:+.1f} m/s: {d['direction_deg']:+.1f}°" for d in r["drives"])
            )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    print(report(tag_calibration(FieldLog(args.log))))


if __name__ == "__main__":
    main()
