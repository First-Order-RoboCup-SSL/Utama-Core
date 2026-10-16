"""How fast the keeper and the nearest defender react to a shot: time from the kick until their
velocity changes, and until they move toward the shot's line.

Velocity comes from positions by central difference over `VEL_SPAN_FRAMES` frames either side,
for real logs and our sim alike (the real tracker's own velocity lags differently). The
baseline is the robot's mean velocity over `BASELINE_S` before the kick, so a robot already
moving is not counted as reacting. A reaction is the first frame whose velocity differs from
the baseline by `REACT_DV_MPS` (`*_react_s`), or whose change points toward the shot line by
that much (`*_toward_s`, only for a robot at least `ON_LINE_M` off the line).

The search runs until the ball passes the robot along the shot, `MAX_S` after the kick, or the
next shot, whichever is first; that length is `*_window_s`. A shot that gives no time to react
says nothing about reaction time, so the scorecard only uses shots with a long enough window.

Caveats: the real tracker filters positions, which delays an onset by a few tens of
milliseconds; the central difference looks `VEL_SPAN_FRAMES` frames ahead, which advances it by
about as much on both sources. The kick time is the shot detector's, a frame or two after
contact on real logs. Read real and sim against each other, not as absolute latencies.
"""

from __future__ import annotations

import bisect
import math
from typing import Optional

from utama_core.analysis.realism.shots import HALF_LEN, keeper, on_field, other, team

VEL_SPAN_FRAMES = 2
BASELINE_S = 0.05
REACT_DV_MPS = 0.25
ON_LINE_M = 0.05
MAX_S = 1.0
DEFENDER_LINE_M = 1.0  # the nearest field defender this close to the shot segment


def _goal_sign(frame, side):
    friendly_goal = -1.0 if frame.my_team_is_right else 1.0
    return friendly_goal if side == "friendly" else -friendly_goal


def _project(px, py, ax, ay, bx, by):
    """(t along a->b in [0, 1] unclamped, distance from the line, unit normal from p to it)."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    cx, cy = ax + t * dx, ay + t * dy
    d = math.hypot(cx - px, cy - py)
    n = ((cx - px) / d, (cy - py) / d) if d > 1e-9 else (0.0, 0.0)
    return t, d, n


def _positions(frames, lo, hi, side, rid):
    out = []
    for f in frames[lo:hi]:
        r = team(f, side).get(rid)
        if r is not None and on_field(r):
            out.append((f.ts, r.p.x, r.p.y))
    return out


def _velocities(track):
    k = VEL_SPAN_FRAMES
    out = []
    for i in range(k, len(track) - k):
        t0, x0, y0 = track[i - k]
        t1, x1, y1 = track[i + k]
        if t1 > t0:
            out.append((track[i][0], (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)))
    return out


def onsets(track, t_kick, t_end, normal) -> tuple[Optional[float], Optional[float]]:
    """(seconds to any velocity change, seconds to a change toward the line) in [t_kick, t_end]."""
    vel = _velocities(track)
    base = [(vx, vy) for t, vx, vy in vel if t_kick - BASELINE_S <= t <= t_kick]
    if not base:
        return None, None
    bx = sum(v[0] for v in base) / len(base)
    by = sum(v[1] for v in base) / len(base)
    react = toward = None
    for t, vx, vy in vel:
        if t <= t_kick or t > t_end:
            continue
        dvx, dvy = vx - bx, vy - by
        if react is None and math.hypot(dvx, dvy) >= REACT_DV_MPS:
            react = round(t - t_kick, 3)
        if toward is None and normal is not None and dvx * normal[0] + dvy * normal[1] >= REACT_DV_MPS:
            toward = round(t - t_kick, 3)
        if react is not None and (toward is not None or normal is None):
            break
    return react, toward


def measure(frames, shots: list[dict]) -> None:
    """Add `keeper_*` and `defender_*` reaction fields to each shot, in place."""
    ts = [f.ts for f in frames]
    times = sorted(s["t"] for s in shots)
    for s in shots:
        i = bisect.bisect_left(ts, s["t"])
        if i >= len(frames):
            continue
        f0 = frames[i]
        defending = other(s["side"])
        g = _goal_sign(f0, s["side"])
        ox, oy = s["raw_x"], s["raw_y"]
        gx, gy = g * HALF_LEN, s["y_at_goal"] * g
        if math.hypot(gx - ox, gy - oy) < 1e-6:
            continue
        nxt = next((t for t in times if t > s["t"] + 1e-6), math.inf)
        kid, kp = keeper(f0, defending)
        robots = []
        if kp is not None and on_field(kp) and abs(kp.p.x - gx) < 2.0:
            robots.append(("keeper", kid, kp))
        field = []
        for rid, r in team(f0, defending).items():
            if rid == kid or not on_field(r):
                continue
            t_along, d, _ = _project(r.p.x, r.p.y, ox, oy, gx, gy)
            if 0.0 < t_along < 1.0 and d <= DEFENDER_LINE_M:
                field.append((d, rid, r))
        if field:
            _, rid, r = min(field, key=lambda x: x[0])
            robots.append(("defender", rid, r))
        for role, rid, r in robots:
            t_along, d, n = _project(r.p.x, r.p.y, ox, oy, gx, gy)
            t_end = min(s["t"] + MAX_S, nxt)
            for f in frames[i:]:  # until the ball passes the robot along the shot
                if f.ts > t_end:
                    break
                b = f.ball
                if b is not None and not math.isnan(b.p.x):
                    if _project(b.p.x, b.p.y, ox, oy, gx, gy)[0] >= min(t_along, 1.0):
                        t_end = f.ts
                        break
            lo = bisect.bisect_left(ts, s["t"] - BASELINE_S - 0.1)
            hi = bisect.bisect_right(ts, t_end + 0.1)
            track = _positions(frames, lo, hi, defending, rid)
            react, toward = onsets(track, s["t"], t_end, n if d >= ON_LINE_M else None)
            s[f"{role}_react_s"] = react
            s[f"{role}_toward_s"] = toward
            s[f"{role}_window_s"] = round(t_end - s["t"], 3)
            s[f"{role}_line_m"] = round(d, 3)
