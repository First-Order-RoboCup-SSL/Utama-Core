"""Physics parameter ranges from replays: ball deceleration, robot motion limits and dropped
detections, for converted official logs and our own matches alike.

    pixi run python -m utama_core.analysis.physics_ranges --real ~/dev/ssl/official-logs/2025 \
        --sim replays/tournament_<id> --out ranges.jsonl

The sim-realism plan's Phase 1 (`docs/sim_realism.md`): the ranges a randomized sim draws from.
Each match gives per-match values (one carpet, one set of robots); the summary reports the spread
between matches, robots or teams ("between") and inside one match ("within"), with 95% bootstrap
intervals over matches for league medians.

Only live play counts (`chances.LIVE`). A real replay's robots and ball come from an autoref's
tracker, a filter: peaks (speed, acceleration, turn rate) come out smoothed, so they are lower
bounds, and the tracker bridges short occlusions itself, so dropped detections are the ones it
gave up on. Real velocities are the tracker's own; accelerations, ball deceleration and turn
rates are central differences over `DIFF_TICKS` either side, the same on both sides.

**Ball deceleration.** A tick is *free* when the ball is in play on the ground (`z` below
`GROUND_Z_M`), seen, more than `FREE_ROBOT_M` from every robot and `WALL_MARGIN_M` inside the
field lines (walls stand beyond them), and moving faster than `MIN_SPEED_MPS`. Its deceleration
is the drop in ball speed over the window, from the velocity and, separately, from positions
(independent of the tracker's velocity model). A *kick* is a ball velocity change of at least
`KICK_DV_MPS` within `KICK_WINDOW_S` to a speed of at least `KICK_SPEED_MPS` (as in the shot
detector). Deceleration is reported by ball speed (constant deceleration means rolling friction;
growing with speed means a velocity-proportional damping) and by time since the last kick (a
sliding phase shows as higher deceleration just after it). A *rolling segment* is a run of free
ticks at least `SLIDE_S` after the last kick and at least `MIN_SEGMENT_S` long; a straight line
fitted to its speed gives one deceleration.

**Robot motion**, per robot with at least `MIN_ROBOT_LIVE_S` of live play: 95th and 99th
percentile speed, acceleration (along the direction of travel, speeding up), braking (along it,
slowing down) and turn rate. Speeds above `GLITCH_SPEED_MPS` (tracker id swaps) are dropped. The
summary keeps robots whose 95th-percentile speed is at least `ACTIVE_SPEED_P95_MPS`: keepers and
the idle robots some real teams leave on the field would otherwise set the range.

**Dropped detections.** A real replay holds the ball at its last position with zero velocity
while the tracker sees none (`ssl_log`); those live ticks are the ball's dropouts. A robot's
dropout is a gap of at most `MAX_ROBOT_GAP_S` with the robot present on both sides, while other
robots are present (all robots missing at once is a tracker outage, counted apart). The sim has
no vision model in our matches (`rsim_noise`/`rsim_vanishing` off), so its values are zero by
construction; the ball held at rest there is not a dropout, so sim ball dropouts are not counted.
The summary leaves out matches whose ball is unseen more than `BROKEN_BALL_SHARE` of live play.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from utama_core.analysis.chances import LIVE
from utama_core.replay.columnar_reader import ColumnarReplay, load_columnar_replay

TICK_S = 1 / 60
DIFF_TICKS = 3  # central differences over +-3 ticks (0.1 s), as the log converter's
GROUND_Z_M = 0.05
FREE_ROBOT_M = 0.3
WALL_MARGIN_M = 0.15
MIN_SPEED_MPS = 0.3
KICK_DV_MPS = 1.5
KICK_WINDOW_S = 0.2
KICK_SPEED_MPS = 2.0
SLIDE_S = 0.5  # real balls decelerate several times faster for about 0.5 s after a kick (sliding)
MIN_SEGMENT_S = 0.3
MIN_SEGMENT_START_MPS = 0.5
MIN_ROBOT_LIVE_S = 20.0
MIN_MOVING_MPS = 0.2  # below this a robot's direction of travel is noise
GLITCH_SPEED_MPS = 6.0
MAX_ROBOT_GAP_S = 3.0
ACTIVE_SPEED_P95_MPS = 0.5  # summary: robots slower than this are keepers or idle robots
BROKEN_BALL_SHARE = 0.2  # summary: a match whose ball is unseen this often has a broken tracker
DEFAULT_FIELD_M = (9.0, 6.0)  # Division B, for replays without a source.json

SPEED_BINS = (0.3, 0.7, 1.5, 3.0, 5.0, 10.0)
SINCE_KICK_BINS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf)
_LIVE_IDS = [c.value for c in LIVE]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """The [start, end) index ranges where `mask` is True."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def window_all(mask: np.ndarray, k: int) -> np.ndarray:
    """True at t where `mask` holds on every tick of [t - k, t + k]."""
    out = np.zeros(len(mask), dtype=bool)
    if len(mask) <= 2 * k:
        return out
    c = np.r_[0, np.cumsum(~np.asarray(mask, dtype=bool))]
    out[k:-k] = (c[2 * k + 1 :] - c[: -2 * k - 1]) == 0
    return out


def central_difference(x: np.ndarray, k: int = DIFF_TICKS, dt: float = TICK_S) -> np.ndarray:
    """d/dt along axis 0 over +-k ticks; NaN at the ends."""
    out = np.full(x.shape, np.nan)
    if len(x) > 2 * k:
        out[k:-k] = (x[2 * k :] - x[: -2 * k]) / (2 * k * dt)
    return out


def kick_ticks(ball_v: np.ndarray, dt: float = TICK_S) -> np.ndarray:
    """First tick of each run where the ball's velocity changed by `KICK_DV_MPS` within
    `KICK_WINDOW_S` and its speed reached `KICK_SPEED_MPS`."""
    lag = max(1, round(KICK_WINDOW_S / dt))
    v = ball_v[:, :2]
    dv = np.zeros(len(v))
    dv[lag:] = np.hypot(*(v[lag:] - v[:-lag]).T)
    hit = (dv >= KICK_DV_MPS) & (np.hypot(*v.T) >= KICK_SPEED_MPS)
    return np.array([s for s, _ in runs(hit)], dtype=int)


def since_last(events: np.ndarray, n: int, dt: float = TICK_S) -> np.ndarray:
    """Seconds since the latest event at or before each tick; inf before the first."""
    last = np.full(n, -1)
    last[events] = events
    last = np.maximum.accumulate(last)
    return np.where(last >= 0, (np.arange(n) - last) * dt, np.inf)


def fit_segment(ts: np.ndarray, speed: np.ndarray) -> tuple[float, float]:
    """Deceleration (minus the slope of a straight line through speed against time) and R²."""
    slope, intercept = np.polyfit(ts, speed, 1)
    resid = speed - (slope * ts + intercept)
    total = np.sum((speed - speed.mean()) ** 2)
    return float(-slope), float(1 - np.sum(resid**2) / total) if total > 0 else 0.0


def _field(path: Path) -> tuple[float, float]:
    source = path.with_name(path.name[: -len(".npz")] + ".source.json")
    if source.exists():
        field = json.loads(source.read_text()).get("field") or {}
        if "field_length" in field:
            return float(field["field_length"]), float(field["field_width"])
    return DEFAULT_FIELD_M


def _teams(path: Path, replay: ColumnarReplay) -> tuple[str, str]:
    """Names for the friendly and enemy columns: an official log's teams, else our configs."""
    source = path.with_name(path.name[: -len(".npz")] + ".source.json")
    if source.exists():
        s = json.loads(source.read_text())
        yellow, blue = s.get("yellow", "yellow"), s.get("blue", "blue")
        return (yellow, blue) if replay.my_team_is_yellow else (blue, yellow)
    a, _, b = path.stem.partition("_vs_")
    return a, b or "enemy"


def ball_seen(ball_p: np.ndarray, ball_v: np.ndarray) -> np.ndarray:
    """False where a converted log holds the ball (unseen): zero velocity, position unchanged."""
    held = np.all(ball_v == 0.0, axis=1) & np.r_[False, np.all(ball_p[1:] == ball_p[:-1], axis=1)]
    return ~held


def free_ball(
    ball_p: np.ndarray, robot_p: np.ndarray, field: tuple[float, float], seen: Optional[np.ndarray] = None
) -> np.ndarray:
    """Ticks where the ball is on the ground, inside the field lines by `WALL_MARGIN_M` and more
    than `FREE_ROBOT_M` from every robot. `robot_p` is (ticks, robots, 2), NaN where absent."""
    gap = robot_p - ball_p[:, None, :2]
    dist = np.hypot(gap[..., 0], gap[..., 1])
    nearest = np.min(np.where(np.isnan(dist), np.inf, dist), axis=1) if dist.shape[1] else np.full(len(ball_p), np.inf)
    inside = (np.abs(ball_p[:, 0]) < field[0] / 2 - WALL_MARGIN_M) & (
        np.abs(ball_p[:, 1]) < field[1] / 2 - WALL_MARGIN_M
    )
    out = (ball_p[:, 2] < GROUND_Z_M) & inside & (nearest > FREE_ROBOT_M)
    return out if seen is None else out & seen


def ball_deceleration(
    ts: np.ndarray, ball_p: np.ndarray, ball_v: np.ndarray, free: np.ndarray, live: np.ndarray
) -> dict:
    """Deceleration samples binned by speed and by time since the last kick, and rolling segments."""
    k = DIFF_TICKS
    dt = TICK_S
    speed = np.hypot(ball_v[:, 0], ball_v[:, 1])
    base = free & live & (speed > MIN_SPEED_MPS)
    decel_v = -central_difference(speed, k, dt)
    pos_speed = np.hypot(*central_difference(ball_p[:, :2], k, dt).T)
    decel_p = -central_difference(pos_speed, k, dt)
    ok_v = window_all(base, k) & np.isfinite(decel_v)
    ok_p = window_all(base, 2 * k) & np.isfinite(decel_p)
    since = since_last(kick_ticks(ball_v, dt), len(ts), dt)

    def binned(values, ok, key, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (key >= lo) & (key < hi)
            rows.append(
                {
                    "lo": lo,
                    "hi": None if np.isinf(hi) else hi,
                    "n": int(m.sum()),
                    "median": float(np.median(values[m])) if m.any() else None,
                }
            )
        return rows

    rolling = base & (since >= SLIDE_S)
    segments = []
    for s, e in runs(rolling):
        if (e - s) * dt < MIN_SEGMENT_S or speed[s] < MIN_SEGMENT_START_MPS:
            continue
        decel, r2 = fit_segment(ts[s:e] - ts[s], speed[s:e])
        segments.append(
            {
                "start_mps": round(float(speed[s]), 3),
                "end_mps": round(float(speed[e - 1]), 3),
                "duration_s": round((e - s) * dt, 3),
                "decel": round(decel, 4),
                "r2": round(r2, 3),
            }
        )
    return {
        "by_speed_vel": binned(decel_v, ok_v, speed, SPEED_BINS),
        "by_speed_pos": binned(decel_p, ok_p, pos_speed, SPEED_BINS),
        "by_since_kick_vel": binned(decel_v, ok_v, since, SINCE_KICK_BINS),
        "by_since_kick_pos": binned(decel_p, ok_p, since, SINCE_KICK_BINS),
        "segments": segments,
    }


def robot_motion(
    robot_p: np.ndarray, robot_v: np.ndarray, orientation: np.ndarray, live: np.ndarray
) -> list[Optional[dict]]:
    """Per robot column: speed, acceleration, braking and turn-rate percentiles in live play
    (None for a robot with less than `MIN_ROBOT_LIVE_S`)."""
    k, dt = DIFF_TICKS, TICK_S
    out = []
    for j in range(robot_p.shape[1]):
        present = ~np.isnan(robot_p[:, j, 0])
        v = robot_v[:, j]
        speed = np.hypot(v[:, 0], v[:, 1])
        sane = present & (np.nan_to_num(speed, nan=np.inf) <= GLITCH_SPEED_MPS)
        ok = window_all(sane & live, k)
        if ok.sum() * dt < MIN_ROBOT_LIVE_S:
            out.append(None)
            continue
        a = central_difference(np.where(sane[:, None], v, np.nan), k, dt)
        moving = ok & (speed > MIN_MOVING_MPS) & np.isfinite(a[:, 0])
        along = np.einsum("ij,ij->i", a[moving], v[moving]) / speed[moving]
        theta = orientation[:, j].copy()
        omega = np.full(len(theta), np.nan)
        for s, e in runs(present):
            if e - s > 2 * k:
                omega[s:e] = central_difference(np.unwrap(theta[s:e]), k, dt)
        turn = np.abs(omega[ok & np.isfinite(omega)])

        def pct(x, q):
            return round(float(np.percentile(x, q)), 3) if len(x) else None

        out.append(
            {
                "live_s": round(float(ok.sum() * dt), 1),
                "speed_p95": pct(speed[ok], 95),
                "speed_p99": pct(speed[ok], 99),
                "accel_p95": pct(along[along > 0], 95),
                "brake_p95": pct(-along[along < 0], 95),
                "turn_p95": pct(turn, 95),
            }
        )
    return out


def dropouts(ball_seen_mask: Optional[np.ndarray], robot_p: np.ndarray, live: np.ndarray) -> dict:
    """Ball and robot dropouts in live play. `ball_seen_mask` None: the ball is not counted."""
    dt = TICK_S
    live_s = float(live.sum() * dt)
    out: dict = {"live_s": round(live_s, 1)}
    if ball_seen_mask is not None:
        lengths = [(e - s) * dt for s, e in runs(~ball_seen_mask & live)]
        out["ball"] = {
            "share": round(sum(lengths) / live_s, 5) if live_s else None,
            "per_live_min": round(len(lengths) / live_s * 60, 3) if live_s else None,
            "run_s": lengths,
        }
    present = ~np.isnan(robot_p[..., 0])
    outage = live & ~present.any(axis=1)
    out["tracker_outage_s"] = round(float(outage.sum() * dt), 2)
    gap_s, gaps, present_s = 0.0, [], 0.0
    for j in range(robot_p.shape[1]):
        p = present[:, j]
        present_s += float((p & live).sum() * dt)
        for s, e in runs(~p):
            if s == 0 or e == len(p) or (e - s) * dt > MAX_ROBOT_GAP_S or outage[s:e].any() or not live[s:e].all():
                continue
            gaps.append((e - s) * dt)
            gap_s += (e - s) * dt
    out["robot"] = {"share": round(gap_s / present_s, 5) if present_s else None, "run_s": gaps}
    return out


def analyse(path_str: str, real: bool) -> dict:
    """Everything for one replay; JSON-serializable."""
    path = Path(path_str)
    try:
        r = load_columnar_replay(path)
        live = np.isin(r.referee_command, _LIVE_IDS)
        field = _field(path)
        robot_p = np.concatenate([r.friendly_p, r.enemy_p], axis=1)
        seen = ball_seen(r.ball_p, r.ball_v) if real else None
        free = free_ball(r.ball_p, robot_p, field, seen)
        names = _teams(path, r)
        robots = []
        for team, p, v, o, ids in (
            (names[0], r.friendly_p, r.friendly_v, r.friendly_orientation, r.friendly_ids),
            (names[1], r.enemy_p, r.enemy_v, r.enemy_orientation, r.enemy_ids),
        ):
            for rid, m in zip(ids.tolist(), robot_motion(p, v, o, live)):
                if m is not None:
                    robots.append({"team": team, "id": int(rid), **m})
        return {
            "match": path.name[: -len(".npz")],
            "real": real,
            "field": field,
            "ball": ball_deceleration(r.ts, r.ball_p, r.ball_v, free, live),
            "robots": robots,
            "dropouts": dropouts(seen, robot_p, live),
        }
    except Exception as e:  # noqa: BLE001 - one bad replay must not stop a corpus pass
        return {"match": path.name, "real": real, "error": repr(e)}


# ---------------------------------------------------------------- summary


def _boot_median(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    meds = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _spread(values: list[float]) -> str:
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(x):
        return "n/a"
    lo, hi = _boot_median(list(x)) if len(x) > 1 else (x[0], x[0])
    q1, q3 = np.percentile(x, [25, 75])
    return (
        f"{np.median(x):.2f} [{lo:.2f}, {hi:.2f}] IQR {q1:.2f}-{q3:.2f} range {x.min():.2f}-{x.max():.2f} (n={len(x)})"
    )


def _pooled_bins(matches: list[dict], key: str) -> list[str]:
    """Median of the per-match medians in each bin, weighted by nothing (one match one vote)."""
    rows = matches[0]["ball"][key]
    out = []
    for i, row in enumerate(rows):
        meds = [m["ball"][key][i]["median"] for m in matches if m["ball"][key][i]["n"] >= 10]
        n = sum(m["ball"][key][i]["n"] for m in matches)
        hi = "inf" if row["hi"] is None else f"{row['hi']:g}"
        out.append(f"  [{row['lo']:g}, {hi}): {_spread(meds)}, ticks={n}")
    return out


def summarise(matches: list[dict]) -> str:
    lines = []
    ok = [m for m in matches if "error" not in m]
    for m in matches:
        if "error" in m:
            lines.append(f"skipped {m['match']}: {m['error']}")
    for label, real in (("REAL", True), ("SIM", False)):
        group = [m for m in ok if m["real"] == real]
        if not group:
            continue
        lines.append(f"\n=== {label}: {len(group)} matches ===")
        lines.append("ball deceleration, m/s² (median of per-match medians [95% CI], IQR and range across matches)")
        for key in ("by_speed_vel", "by_speed_pos", "by_since_kick_vel", "by_since_kick_pos"):
            lines.append(f" {key}:")
            lines += _pooled_bins(group, key)
        per_match, within, n_seg = [], [], 0
        for m in group:
            d = [s["decel"] for s in m["ball"]["segments"] if s["r2"] >= 0.5]
            n_seg += len(d)
            if len(d) >= 5:
                per_match.append(float(np.median(d)))
                q1, q3 = np.percentile(d, [25, 75])
                within.append(float(q3 - q1))
        lines.append(f" rolling segments (R² >= 0.5, {n_seg} in all), per-match median (between): {_spread(per_match)}")
        lines.append(f" rolling segments, within-match IQR: {_spread(within)}")

        every = [r for m in group for r in m["robots"]]
        robots = [r for r in every if (r["speed_p95"] or 0) >= ACTIVE_SPEED_P95_MPS]
        lines.append(
            f"robot motion, per active robot ({len(robots)} of {len(every)} robot-matches): "
            "median [95% CI over robots] IQR range"
        )
        for key, unit in (
            ("speed_p95", "m/s"),
            ("speed_p99", "m/s"),
            ("accel_p95", "m/s²"),
            ("brake_p95", "m/s²"),
            ("turn_p95", "rad/s"),
        ):
            lines.append(f"  {key} ({unit}): {_spread([r[key] for r in robots])}")
            teams: dict[str, list[float]] = {}
            for r in robots:
                if r[key] is not None:
                    teams.setdefault(r["team"], []).append(r[key])
            team_meds = [float(np.median(v)) for v in teams.values() if len(v) >= 3]
            lines.append(f"    team medians (between teams): {_spread(team_meds)}")

        drops = [m["dropouts"] for m in group]
        if real:
            broken = [m["match"] for m in group if m["dropouts"]["ball"]["share"] > BROKEN_BALL_SHARE]
            drops = [d for d in drops if d["ball"]["share"] <= BROKEN_BALL_SHARE]
            if broken:
                lines.append(f"left out of dropouts, ball unseen > {BROKEN_BALL_SHARE:.0%}: {', '.join(broken)}")
            lines.append("dropped detections (live play)")
            lines.append(f"  ball unseen share per match: {_spread([d['ball']['share'] for d in drops])}")
            lines.append(f"  ball dropouts per live minute: {_spread([d['ball']['per_live_min'] for d in drops])}")
            runs_s = [x for d in drops for x in d["ball"]["run_s"]]
            if runs_s:
                q = np.percentile(runs_s, [50, 90, 99])
                lines.append(
                    f"  ball dropout length s: median {q[0]:.3f} p90 {q[1]:.3f} p99 {q[2]:.3f} max {max(runs_s):.2f} (n={len(runs_s)})"
                )
        lines.append(f"  robot gap share per match: {_spread([d['robot']['share'] for d in drops])}")
        gaps = [x for d in drops for x in d["robot"]["run_s"]]
        if gaps:
            q = np.percentile(gaps, [50, 90, 99])
            lines.append(f"  robot gap length s: median {q[0]:.3f} p90 {q[1]:.3f} p99 {q[2]:.3f} (n={len(gaps)})")
        lines.append(f"  tracker outage s per match: {_spread([d['tracker_outage_s'] for d in drops])}")
    return "\n".join(lines)


def _replays(dirs: list[str]) -> list[str]:
    out = []
    for d in dirs:
        p = Path(d).expanduser()
        out += [str(q) for q in sorted(p.rglob("*.npz")) if "match_cache" not in q.parts] if p.is_dir() else [str(p)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real", nargs="*", default=[], help="converted official logs (dirs or .npz)")
    parser.add_argument("--sim", nargs="*", default=[], help="our replays (dirs or .npz)")
    parser.add_argument("--out", help="write one JSON line per match here")
    parser.add_argument("--from-jsonl", nargs="*", default=[], help="summarise earlier --out files instead")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    matches = [json.loads(line) for p in args.from_jsonl for line in open(p)]
    jobs = [(p, True) for p in _replays(args.real)] + [(p, False) for p in _replays(args.sim)]
    if jobs:
        out = open(args.out, "w") if args.out else None
        with ProcessPoolExecutor(max(1, args.workers)) as ex:
            for i, m in enumerate(ex.map(analyse, *zip(*jobs))):
                matches.append(m)
                if out:
                    out.write(json.dumps(m) + "\n")
                print(f"[{i + 1}/{len(jobs)}] {m['match']} {m.get('error', '')}", flush=True)
        if out:
            out.close()
    print(summarise(matches))


if __name__ == "__main__":
    main()
