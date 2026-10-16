"""Pass accuracy, by proxy: the logs record no intended receiver, so the receiver is inferred.

Passes are softer than shots, so they get their own kick detector: in live play, the ball's
speed rises by `KICK_DV_MPS` within `shots.KICK_WINDOW_S` to at least `KICK_MPS`, with a robot
(the kicker) within `KICKER_M` of where the ball was. Kicks that are shots (`shots.scan`) are
left out. The kick's direction is the ball's velocity `shots.SETTLE_S` after it. Its **target**
is the kicker's teammate nearest the kick's line among those ahead of the ball by `AHEAD_M`,
and at most `LINE_M` off the line; a kick with no such teammate (a clearance, a dribble push)
is not a pass. Per pass:

- `offset_m`: that teammate's distance from the line at the kick: how far off the kick was,
  plus however far the receiver still had to move;
- `completed`: the next robot other than the kicker within `shot_outcomes.TOUCH_M` of the
  ball, at most `FOLLOW_S` later and before the next kick, is a teammate.

Both sources go through the same inference, so read real against sim, not as an aim error.
"""

from __future__ import annotations

import bisect
import math

from utama_core.analysis.chances import LIVE
from utama_core.analysis.realism.shot_outcomes import SKIP_S, TOUCH_M
from utama_core.analysis.realism.shots import (
    KICK_WINDOW_S,
    REFRACTORY_S,
    SETTLE_S,
    on_field,
    team,
)

KICK_MPS = 1.0
KICK_DV_MPS = 0.8
KICKER_M = 0.3
AHEAD_M = (0.5, 6.0)
LINE_M = 1.0
FOLLOW_S = 3.0


def _seen(ball) -> bool:
    return ball is not None and not math.isnan(ball.p.x)


def _kicker(frame, x, y):
    best = None
    for side in ("friendly", "enemy"):
        for rid, r in team(frame, side).items():
            if on_field(r):
                d = math.hypot(r.p.x - x, r.p.y - y)
                if d <= KICKER_M and (best is None or d < best[0]):
                    best = (d, side, rid)
    return None if best is None else best[1:]


def kicks(frames) -> list[tuple[int, float, float]]:
    """(frame index, x, y where the ball was) per live kick, `REFRACTORY_S` apart."""
    out, history, last = [], [], -math.inf
    for i, f in enumerate(frames):
        b = f.ball
        if not _seen(b):
            continue
        history.append(i)
        while frames[i].ts - frames[history[0]].ts > KICK_WINDOW_S:
            history.pop(0)
        live = f.referee is not None and f.referee.referee_command in LIVE
        before = frames[history[0]].ball
        speed = math.hypot(b.v.x, b.v.y)
        rise = speed - math.hypot(before.v.x, before.v.y)
        if live and speed >= KICK_MPS and rise >= KICK_DV_MPS and f.ts - last >= REFRACTORY_S:
            last = f.ts
            out.append((i, before.p.x, before.p.y))
    return out


def follow(frames, shot_times) -> list[dict]:
    """One record per pass: kicker side, `offset_m`, `completed`, ball speed."""
    ts = [f.ts for f in frames]
    found = kicks(frames)
    starts = [frames[i].ts for i, _, _ in found]
    shot_times = sorted(shot_times)
    out = []
    for n, (i, x, y) in enumerate(found):
        # a shot detected within a kick window of this kick is the same kick
        k = bisect.bisect_left(shot_times, ts[i] - KICK_WINDOW_S)
        if k < len(shot_times) and shot_times[k] <= ts[i] + KICK_WINDOW_S:
            continue
        j = bisect.bisect_left(ts, ts[i] + SETTLE_S)
        if j >= len(frames) or not _seen(frames[j].ball):
            continue
        b = frames[j].ball
        speed = math.hypot(b.v.x, b.v.y)
        if speed < KICK_MPS:
            continue
        ux, uy = b.v.x / speed, b.v.y / speed
        kicker = _kicker(frames[i], x, y)
        if kicker is None:
            continue
        side, kid = kicker
        offsets = []
        for rid, r in team(frames[i], side).items():
            if rid == kid or not on_field(r):
                continue
            rx, ry = r.p.x - x, r.p.y - y
            along = rx * ux + ry * uy
            off = abs(rx * uy - ry * ux)
            if AHEAD_M[0] <= along <= AHEAD_M[1] and off <= LINE_M:
                offsets.append(off)
        if not offsets:
            continue
        nxt = starts[n + 1] if n + 1 < len(starts) else math.inf
        completed = False
        for f in frames[i:]:
            dt = f.ts - ts[i]
            if dt > FOLLOW_S or f.ts >= nxt:
                break
            fb = f.ball
            if dt <= SKIP_S or not _seen(fb):
                continue
            hit = next(
                (
                    (s, rid)
                    for s in ("friendly", "enemy")
                    for rid, r in team(f, s).items()
                    if not (s == side and rid == kid)
                    and on_field(r)
                    and math.hypot(r.p.x - fb.p.x, r.p.y - fb.p.y) <= TOUCH_M
                ),
                None,
            )
            if hit:
                completed = hit[0] == side
                break
        out.append({"side": side, "offset_m": round(min(offsets), 3), "completed": completed, "speed": round(speed, 2)})
    return out
