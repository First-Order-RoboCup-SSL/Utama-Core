"""World physics measured from field logs, for the sim's settings: loop delay, ball rolling
deceleration, and detection noise at rest for the ball and the robots.

Only the sessions and measurements allowed in `tools/field_sessions.json` are read. Each log's
values are grouped by (day, radio) and printed as a median with its middle half and count, then
checked for agreement across the groups: a setting is only trusted if the days repeat it.

    pixi run python tools/field_physics.py [--logs replays/field] [--workers 2] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis import robot_calibration as rc  # noqa: E402

REGISTER = Path(__file__).with_name("field_sessions.json")
MEASUREMENTS = ("loop_delay", "ball_roll", "ball_noise", "robot_noise")
UNITS = {"loop_delay": "ms", "ball_roll": "m/s^2", "ball_noise": "mm", "robot_noise": "mm"}
ROLL_MAX_RMS_MM = 12.0  # a roll fit this noisy is not a clean roll
ROLL_MIN_DECEL = 0.0
ROLL_MIN_SECS = 0.6
STEP_MIN_CMD = 0.3  # commanded speed (m/s or rad/s) for a step to count as a delay measurement
STEP_AXES = ("forward", "left")
FIELD_BOUNDS = {"top_left": [-1.5, 1.125], "bottom_right": [1.5, -1.125]}
ROLL_MIN_SPEED = 0.8  # m/s, ball speed of a kick
ROLL_EDGE_M = 0.1  # the fit stops this far from the field edge
ROLL_ROBOT_M = 0.25  # and this far from any robot


def load_register(path: Path = REGISTER) -> dict:
    """The field sessions register: which days may be used for which measurements."""
    with open(path) as f:
        return json.load(f)


def session_of(log_name: str) -> str:
    """The day (first 8 characters of the file name), e.g. '20261028' for '20261028_...log.gz'."""
    return Path(log_name).name[:8]


def allowed(register: dict, day: str, measurement: str) -> bool:
    """True only if the day is registered and lists the measurement in its `use`."""
    session = register.get("sessions", {}).get(day)
    if session is None:
        return False
    return measurement in session.get("use", [])


def ball_rolls(log, robot_kicks: list[dict], bounds) -> list[dict]:
    """Rolling deceleration after clean kicks, single camera, away from robots and field edge.

    Each kick's ball path is fitted with a quadratic in distance travelled; the fit is kept only
    if it is clean (rms < ROLL_MAX_RMS_MM), decelerating and at least ROLL_MIN_SECS long."""
    if len(log.balls) < 30:
        return []
    b = rc._main_camera(np.column_stack([log.balls[:, :2], log.balls[:, 2:], np.zeros(len(log.balls))])).astype(float)
    out = []
    (xl, yt), (xr, yb) = bounds
    for k in robot_kicks:
        if k["ball_speed"] < ROLL_MIN_SPEED or k.get("chip"):
            continue
        t0 = k["t"]
        w = (b[:, 0] > t0 + 0.3) & (b[:, 0] < t0 + 5)
        if w.sum() < 25:
            continue
        r = b[w]
        dt = np.diff(r[:, 0])
        cut = np.flatnonzero(dt > 0.1)
        if len(cut):
            r = r[: cut[0] + 1]
        inside = (
            (r[:, 2] > xl + ROLL_EDGE_M)
            & (r[:, 2] < xr - ROLL_EDGE_M)
            & (r[:, 3] > yb + ROLL_EDGE_M)
            & (r[:, 3] < yt - ROLL_EDGE_M)
        )
        # distance from the ball to the nearest robot at each sample
        d = np.full(len(r), np.inf)
        for rows in log.robots.values():
            q = rows[np.argsort(rows[:, 0].astype(float))].astype(float)
            d = np.minimum(
                d,
                np.hypot(
                    np.interp(r[:, 0], q[:, 0], q[:, 2]) - r[:, 2], np.interp(r[:, 0], q[:, 0], q[:, 3]) - r[:, 3]
                ),
            )
        ok = inside & (d > ROLL_ROBOT_M)
        n = int(np.argmin(ok)) if not ok.all() else len(r)
        r = r[:n]
        if len(r) < 25:
            continue
        t = r[:, 0] - r[0, 0]
        s = np.r_[0, np.cumsum(np.hypot(np.diff(r[:, 2]), np.diff(r[:, 3])))]
        moving = np.flatnonzero(np.gradient(s, t) > 0.15)
        if len(moving) < 25:
            continue
        e = moving[-1] + 1
        t, s = t[:e], s[:e]
        if t[-1] < 0.4:
            continue
        c = np.polyfit(t, s, 2)
        row = dict(
            decel=round(float(-2 * c[0]), 3),
            v0=round(float(c[1]), 3),
            secs=round(float(t[-1]), 2),
            rms_mm=round(1000 * float(np.std(s - np.polyval(c, t))), 2),
        )
        if row["rms_mm"] < ROLL_MAX_RMS_MM and row["decel"] > ROLL_MIN_DECEL and row["secs"] >= ROLL_MIN_SECS:
            out.append(row)
    return out


def step_delays_ms(steps: list[dict]) -> list[float]:
    """Each fitted step's delay in ms, for steps moving forward or left at >= STEP_MIN_CMD."""
    out = []
    for s in steps:
        delay = s.get("delay_s")
        if not delay or delay <= 0:
            continue
        if s.get("axis") not in STEP_AXES or abs(s.get("commanded", 0.0)) < STEP_MIN_CMD:
            continue
        out.append(delay * 1000)
    return out


def measure_log(path) -> Optional[dict]:
    """The allowed measurements of one field log, or None if its mode is not real or its day
    allows nothing. Each value is (measurement, radio or None, value)."""
    name = os.path.basename(str(path))
    day = session_of(name)
    register = load_register()
    if not any(allowed(register, day, m) for m in MEASUREMENTS):
        return None
    try:
        log = rc.FieldLog(path)
        if log.info.get("mode") != "real":
            return None
        fb = log.info.get("field_bounds", FIELD_BOUNDS)
        bounds = (fb["top_left"], fb["bottom_right"])
        values: list[tuple] = []
        for r in rc.calibrate(log):
            if "error" in r:
                continue
            fw = r["firmware_id"]
            if allowed(register, day, "loop_delay"):
                values += [("loop_delay", fw, ms) for ms in step_delays_ms(r["steps"])]
            if allowed(register, day, "ball_roll"):
                values += [("ball_roll", fw, roll["decel"]) for roll in ball_rolls(log, r["kicks"], bounds)]
            if allowed(register, day, "robot_noise") and r["vision"].get("position_mm") is not None:
                values.append(("robot_noise", fw, r["vision"]["position_mm"]))
        if allowed(register, day, "ball_noise"):
            pos = rc.ball_noise(log.balls).get("position_mm")
            if pos is not None:
                values.append(("ball_noise", None, pos))
        return {"log": name, "day": day, "values": values}
    except Exception as e:  # one bad log must not lose the run
        return {"log": name, "day": day, "error": str(e), "values": []}


def summarise(rows: list[dict]) -> dict:
    """{measurement: {(day, radio): {median, lo, hi, n}}}: lo and hi are the 25th and 75th percentiles."""
    grouped: dict[str, dict[tuple, list[float]]] = {}
    for row in rows:
        if not row:
            continue
        for measurement, radio, value in row["values"]:
            grouped.setdefault(measurement, {}).setdefault((row["day"], radio), []).append(float(value))
    out: dict = {}
    for measurement, groups in grouped.items():
        out[measurement] = {}
        for key, vals in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
            a = np.array(vals, float)
            out[measurement][key] = {
                "median": round(float(np.median(a)), 3),
                "lo": round(float(np.percentile(a, 25)), 3),
                "hi": round(float(np.percentile(a, 75)), 3),
                "n": len(a),
            }
    return out


def agreement(groups: dict, tolerance: float = 0.25, min_n: int = 5) -> dict:
    """Whether the groups' medians agree: highest - lowest <= tolerance * median of the medians.
    Only groups with at least `min_n` values count. Fewer than two usable groups gives agree None."""
    medians = [g["median"] for g in groups.values() if g["n"] >= min_n]
    if len(medians) < 2:
        return {"medians": [min(medians), max(medians)] if medians else [], "agree": None, "groups_used": len(medians)}
    lowest, highest = min(medians), max(medians)
    return {
        "medians": [lowest, highest],
        "agree": bool(highest - lowest <= tolerance * float(np.median(medians))),
        "groups_used": len(medians),
    }


def _key(day: str, radio) -> str:
    return f"{day}|{'all' if radio is None else radio}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", default="replays/field", help="folder of *.log.gz files")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--json", type=Path, help="write the summary, agreement and errors here")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.logs, "*.log.gz")))
    print(f"{len(files)} logs in {args.logs}")
    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap(measure_log, files, chunksize=1) if r is not None]
    errors = [r for r in rows if "error" in r]
    for r in errors:
        print(f"ERROR {r['log']}: {r['error']}")
    print(f"{len(rows)} logs used, {len(errors)} errored")

    summary = summarise(rows)
    agree_all: dict = {}
    for measurement in MEASUREMENTS:
        groups = summary.get(measurement, {})
        unit = UNITS[measurement]
        print(f"\n{measurement} ({unit})")
        if not groups:
            print("  no values")
            agree_all[measurement] = agreement({})
            continue
        print(f"  {'day':<10}{'radio':<7}{'median':>10}{'middle half':>24}{'n':>7}")
        for (day, radio), g in groups.items():
            radio_s = "all" if radio is None else str(radio)
            middle = "%s to %s" % (g["lo"], g["hi"])
            print(f"  {day:<10}{radio_s:<7}{g['median']:>10}{middle:>24}{g['n']:>7}")
        ag = agreement(groups)
        agree_all[measurement] = ag
        if ag["agree"] is None:
            verdict = f"too few groups to compare ({ag['groups_used']} with n >= 5)"
        elif ag["agree"]:
            verdict = "agree"
        else:
            verdict = "disagree: find the setup difference or use the spread as the range"
        print(f"  agreement: {verdict}")

    if args.json:
        payload = {
            "summary": {m: {_key(d, r): g for (d, r), g in groups.items()} for m, groups in summary.items()},
            "agreement": agree_all,
            "errors": [{"log": r["log"], "error": r["error"]} for r in errors],
        }
        args.json.write_text(json.dumps(payload, indent=1))
        print(f"\nwritten {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
