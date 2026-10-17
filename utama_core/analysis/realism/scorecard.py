"""The realism scorecard: our sim against real Division B matches on the parts that decide goals.

Goal totals can agree for the wrong reasons (2026-10-14: sim keepers save twice as many good
chances while sim shooters almost never miss, and the two cancel), so each row is one
component. A row holds the real value with a 95% bootstrap interval over matches, the sim value
with its own interval, and a pass mark: the sim value inside the real interval. Rows driven by
the referee (live minutes, restarts) are shown but not marked; they differ for referee reasons,
not physics.

Every row is computed over **attempts** (`shot_outcomes.is_attempt`: no pass to a teammate,
at most 6 m out), scored with one xG model fitted on the real attempts, except the pass rows
(`passes.py`). The real side, with the
fitted model, is stored in a reference file (`REFERENCE_PATH`), so the sim side needs no real
logs. See `docs/sim_realism.md`.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from utama_core.analysis.realism import passes, reaction, shot_outcomes, shots, xg

REFERENCE_PATH = Path(__file__).with_name("real_2025_2026.json")
XG_BANDS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 1.01))
TOP_BAND = XG_BANDS[-1]
BOOTSTRAP_N = 1000
KEPT_SPEED_RANGE_M = (3.0, 5.0)  # shots this far out for the ball speed kept 1 m from goal
REFEREE_ROWS = ("live minutes per match", "restarts per match")
LONG_ATTEMPT_M = 4.0
REACT_WINDOW_S = 0.4  # reaction rows use only shots that leave at least this long to react
# Matches whose tracker failed: 2026-07-02 TurtleRabbit vs NAMeC has the ball unseen 57% of the
# time, with a 782 s outage.
EXCLUDED_MATCHES = frozenset({"2026-07-02_18-16_GROUP_PHASE_TurtleRabbit-vs-NAMeC"})


def analyse_replay(path) -> dict:
    """One match's record: shots with features, goal links and outcomes, live time, restarts."""
    from utama_core.replay.replay_player import load_frames_in_range

    path = Path(path)
    stem = path.name[: -len(".npz")] if path.name.endswith(".npz") else path.stem
    try:
        frames = load_frames_in_range(path, 0.0, math.inf)
        scanned = shots.scan(frames)
        links = shots.link_goals(scanned)
        shot_outcomes.follow(frames, scanned["shots"])
        reaction.measure(frames, scanned["shots"])
        pass_list = passes.follow(frames, [s["t"] for s in scanned["shots"]])
        live_s = scanned["live_s"]
        return {
            "match": stem,
            "live_s": round(live_s, 1),
            "goals": scanned["goals"],
            "restarts": scanned["restarts"],
            "robots_live": {s: round(v / live_s, 2) if live_s else None for s, v in scanned["robots_live"].items()},
            **links,
            "shots": scanned["shots"],
            "passes": pass_list,
        }
    except Exception as e:  # noqa: BLE001 - one bad log must not stop a corpus pass
        return {"match": stem, "error": repr(e)}


def usable(records: list[dict]) -> list[dict]:
    """Records without an error, not excluded, with at least `xg.MIN_LIVE_S` of live play, shots
    cut to attempts."""
    out = []
    for r in records:
        if "error" in r or r["live_s"] < xg.MIN_LIVE_S or r.get("match") in EXCLUDED_MATCHES:
            continue
        out.append({**r, "shots": [s for s in r["shots"] if shot_outcomes.is_attempt(s)]})
    return out


def _react(a, role, key):
    """Onsets (inf where none came in time) of `role` on shots leaving at least `REACT_WINDOW_S`
    to react; toward-line onsets only for a robot off the line."""
    out = []
    for s in a:
        window = s.get(f"{role}_window_s")
        if window is None or window < REACT_WINDOW_S:
            continue
        if key == "toward" and s[f"{role}_line_m"] < reaction.ON_LINE_M:
            continue
        v = s.get(f"{role}_{key}_s")
        out.append(math.inf if v is None or v > REACT_WINDOW_S else v)
    return out


def _arr(xs, dtype=float):
    return np.array(list(xs), dtype=dtype)


class _Match:
    """One match's attempts and passes as arrays, so a bootstrap sample is a concatenation."""

    def __init__(self, record: dict, model: xg.Logit):
        a = record["shots"]
        p = record.get("passes", [])
        self.live_s = record["live_s"]
        self.restarts = record.get("restarts", 0)
        lo, hi = KEPT_SPEED_RANGE_M
        self.arrays = {
            "xg": model.predict(np.array([xg.row(s) for s in a])) if a else np.empty(0),
            "scored": _arr(s["scored"] for s in a),
            "outcome": _arr((s["outcome"] for s in a), object),
            "set_piece": _arr((s["context"] != "open_play" for s in a), bool),
            "distance": _arr(s["distance_m"] for s in a),
            "kick": _arr(s["peak"] for s in a),
            "aim_out": _arr(abs(s.get("y_at_goal", 0.0)) > shots.HALF_GOAL for s in a),
            "kept": _arr(
                s["speed_1m"] / s["peak"]
                for s in a
                if s.get("speed_1m") and s["peak"] > shots.KICK_MPS and lo <= s["distance_m"] <= hi
            ),
            "depth": _arr(s["keeper_depth_m"] for s in a if s.get("keeper_depth_m") is not None),
            "miss": _arr(s["miss_m"] for s in a if s.get("miss_m") is not None),
            "keeper_react": _arr(_react(a, "keeper", "react")),
            "keeper_toward": _arr(_react(a, "keeper", "toward")),
            "defender_toward": _arr(_react(a, "defender", "toward")),
            "pass_offset": _arr(x["offset_m"] for x in p),
            "pass_completed": _arr(x["completed"] for x in p),
        }


def _share(outcomes: tuple[str, ...], band: Optional[tuple[float, float]] = None) -> Callable:
    def f(c):
        sel = np.ones(len(c["xg"]), bool) if band is None else (c["xg"] >= band[0]) & (c["xg"] < band[1])
        return float(np.isin(c["outcome"][sel], outcomes).mean()) if sel.any() else math.nan

    return f


def _band_goal_rate(band):
    def f(c):
        sel = (c["xg"] >= band[0]) & (c["xg"] < band[1])
        return float(c["scored"][sel].mean()) if sel.any() else math.nan

    return f


def _median(key):
    return lambda c: float(np.median(c[key])) if len(c[key]) else math.nan


def _mean(key):
    return lambda c: float(np.mean(c[key])) if len(c[key]) else math.nan


def _reacted_share(key):
    return lambda c: float(np.isfinite(c[key]).mean()) if len(c[key]) else math.nan


def _reacted_median(key):
    def f(c):
        v = c[key][np.isfinite(c[key])]
        return float(np.median(v)) if len(v) else math.nan

    return f


def _attempts_per_minute(set_piece):
    return lambda c: float((c["set_piece"] == set_piece).sum()) / (c["live_s"] / 60 * 2)


ROWS: dict[str, Callable] = {
    "attempts per team per live minute": lambda c: len(c["xg"]) / (c["live_s"] / 60 * 2),
    "open-play attempts per team per live minute": _attempts_per_minute(False),
    "set-piece attempts per team per live minute": _attempts_per_minute(True),
    "attempt distance, median (m)": _median("distance"),
    f"attempts beyond {LONG_ATTEMPT_M:g} m, share": lambda c: (
        float((c["distance"] > LONG_ATTEMPT_M).mean()) if len(c["distance"]) else math.nan
    ),
    "real xG per attempt": lambda c: float(c["xg"].mean()),
    "finishing (goals / real xG)": lambda c: float(c["scored"].sum() / c["xg"].sum()),
    **{f"goal rate, xG {lo:g}-{min(hi, 1):g}": _band_goal_rate((lo, hi)) for lo, hi in XG_BANDS},
    "outcome: goal": _share(("goal",)),
    "outcome: keeper": _share(("keeper",)),
    **{f"outcome: keeper, xG {lo:g}-{min(hi, 1):g}": _share(("keeper",), (lo, hi)) for lo, hi in XG_BANDS},
    "outcome: defender": _share(("defender",)),
    "outcome: wide, post or crossed": _share(("wide", "post", "crossed_no_goal")),
    "outcome: wide": _share(("wide",)),
    "outcome: post": _share(("post",)),
    "outcome: crossed, no goal": _share(("crossed_no_goal",)),
    "aim outside the posts, share": _mean("aim_out"),
    "wide miss beyond the post, median (m)": _median("miss"),
    "ball speed kept 1 m from goal (3-5 m shots)": _median("kept"),
    "kick speed, median (m/s)": _median("kick"),
    "kick speed, interquartile range (m/s)": lambda c: (
        float(np.subtract(*np.percentile(c["kick"], [75, 25]))) if len(c["kick"]) else math.nan
    ),
    "keeper depth off the goal line, median (m)": _median("depth"),
    f"keeper velocity change within {REACT_WINDOW_S:g} s, share": _reacted_share("keeper_react"),
    "keeper velocity change, median (s)": _reacted_median("keeper_react"),
    f"keeper moves toward the shot line within {REACT_WINDOW_S:g} s, share": _reacted_share("keeper_toward"),
    "keeper moves toward the shot line, median (s)": _reacted_median("keeper_toward"),
    f"nearest defender moves toward the line within {REACT_WINDOW_S:g} s, share": _reacted_share("defender_toward"),
    "nearest defender moves toward the line, median (s)": _reacted_median("defender_toward"),
    "pass line offset to the nearest teammate ahead, median (m)": _median("pass_offset"),
    "pass completion (next touch a teammate)": _mean("pass_completed"),
    "live minutes per match": lambda c: c["live_s"] / 60 / c["n"],
    "restarts per match": lambda c: c["restarts"] / c["n"],
}


def _pool(ms: list[_Match]) -> dict:
    pooled = {k: np.concatenate([m.arrays[k] for m in ms]) for k in ms[0].arrays} if ms else {}
    pooled.update(
        live_s=sum(m.live_s for m in ms),
        restarts=sum(m.restarts for m in ms),
        n=len(ms),
    )
    return pooled


def rows(records: list[dict], model: xg.Logit, n_boot: int = BOOTSTRAP_N, seed: int = 0) -> dict:
    """{row: {value, lo, hi}} over `records` (already `usable`), 95% intervals by resampling matches."""
    ms = [_Match(r, model) for r in records]
    point = {k: f(_pool(ms)) for k, f in ROWS.items()}
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ROWS}
    for _ in range(n_boot):
        pooled = _pool([ms[i] for i in rng.integers(0, len(ms), len(ms))])
        for k, f in ROWS.items():
            samples[k].append(f(pooled))
    out = {}
    for k, v in point.items():
        with warnings.catch_warnings():  # a row with no data in a resample (an empty xG band)
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(samples[k], [2.5, 97.5])
        out[k] = {"value": v, "lo": float(lo), "hi": float(hi)}
    return out


def reference(real_records: list[dict], sources: list[str]) -> dict:
    """The real side of the scorecard, stored so the sim side needs no real logs."""
    real = usable(real_records)
    X, y, g = xg.design(real)
    model = xg.Logit().fit(X, y)
    return {
        "sources": sources,
        "matches": len(real),
        "attempts": len(y),
        "goals_from_attempts": int(y.sum()),
        "xg_model": model.to_dict(),
        "xg_check": xg.cross_validate(X, y, g),
        "rows": rows(real, model),
    }


def compare(ref: dict, sim_records: list[dict]) -> list[dict]:
    """One line per row: real value and interval, sim value and interval, and whether it passes."""
    sim = usable(sim_records)
    model = xg.Logit.from_dict(ref["xg_model"])
    sim_rows = rows(sim, model)
    out = []
    for k, r in ref["rows"].items():
        s = sim_rows[k]
        passes = None if k in REFEREE_ROWS else bool(r["lo"] <= s["value"] <= r["hi"])
        out.append(
            {"row": k, "real": r, "sim": s, "pass": passes, "ratio": s["value"] / r["value"] if r["value"] else None}
        )
    return out


def format_table(lines: list[dict]) -> str:
    def fmt(x):
        return f"{x['value']:.3g} [{x['lo']:.3g}, {x['hi']:.3g}]"

    width = max(len(line["row"]) for line in lines)
    head = f"{'row':{width}}  {'real [95%]':>24}  {'sim [95%]':>24}  {'sim/real':>8}  pass"
    out = [head, "-" * len(head)]
    for line in lines:
        mark = {True: "yes", False: "NO", None: "-"}[line["pass"]]
        ratio = "" if line["ratio"] is None else f"x{line['ratio']:.2f}"
        out.append(f"{line['row']:{width}}  {fmt(line['real']):>24}  {fmt(line['sim']):>24}  {ratio:>8}  {mark}")
    return "\n".join(out)
