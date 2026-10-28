"""What happened to each shot `shots.scan` found, and which shots count as attempts.

`follow` tracks the ball for up to `FOLLOW_S` after each shot (or until the next shot) and
records the first of:

- `goal`: the shot is marked scored;
- `keeper` / `defender` / `teammate`: a robot comes within `TOUCH_M` of the ball;
- `post`: the ball deflects within `POST_NEAR_M` of a post (its direction turns by
  `POST_TURN_DEG` or its speed drops to `POST_SLOW` of what it was `POST_LOOKBACK_S` earlier,
  with no robot touching), or crosses the goal line within `POST_BAND_M` of a post (ball
  radius plus post thickness: the frame it hit the post);
- `wide`: the ball crosses the goal line outside the mouth;
- `crossed_no_goal`: it crosses inside the mouth but no goal was given;
- `stopped`: the ball slows below `STOP_MPS` short of the goal;
- `open`: none of these within `FOLLOW_S`.

Also measured at the kick: the keeper's distance from the shot line (origin to where the line
meets the goal line), its depth off the goal line and its offset across from the aim point, and
the ball speed when it reaches 1 m from the goal line, for the ball's slow-down. Where the ball
crossed the goal line (`cross_y`, shooter's view) and, for a wide shot, how far outside the post
(`miss_m`).

**Attempts** (`is_attempt`), the default shot definition for the scorecard and the xG fit: a
shot whose next touch is not a teammate (that is a pass) and that starts at most
`MAX_ATTEMPT_M` from the goal (beyond that the sim's on-target kicks are mostly clearances from
its own box down the middle; real teams score only 6% of their goals from there).
"""

from __future__ import annotations

import bisect
import math

from utama_core.analysis.realism.shots import (
    HALF_GOAL,
    HALF_LEN,
    keeper,
    on_field,
    other,
    team,
)

FOLLOW_S = 2.5
TOUCH_M = 0.14  # robot radius 0.09 + ball radius 0.0215 + tracking slack
STOP_MPS = 0.3
SKIP_S = 0.1  # ignore the kicker's own contact just after the kick
MAX_ATTEMPT_M = 6.0
BALL_RADIUS_M = 0.0215
POST_THICKNESS_M = 0.02
POST_BAND_M = BALL_RADIUS_M + POST_THICKNESS_M
POST_NEAR_M = 0.1
POST_TURN_DEG = 30.0
POST_SLOW = 0.6
POST_LOOKBACK_S = 0.05  # the real tracker smooths a bounce over a few frames
OUTCOMES = ("goal", "keeper", "defender", "teammate", "post", "wide", "crossed_no_goal", "stopped", "open")


def _seen(ball) -> bool:
    return ball is not None and not math.isnan(ball.p.x)


def _deflected(v_then, v_now) -> bool:
    s0, s1 = math.hypot(*v_then), math.hypot(*v_now)
    if s0 < STOP_MPS:
        return False
    if s1 < POST_SLOW * s0:
        return True
    cos = (v_then[0] * v_now[0] + v_then[1] * v_now[1]) / (s0 * s1)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > POST_TURN_DEG


def is_attempt(shot: dict) -> bool:
    return shot.get("outcome") != "teammate" and shot["distance_m"] <= MAX_ATTEMPT_M


def _goal_sign(frame, side):
    friendly_goal = -1.0 if frame.my_team_is_right else 1.0
    return friendly_goal if side == "friendly" else -friendly_goal


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def follow(frames, shots: list[dict]) -> None:
    """Add `outcome`, `speed_1m` and the keeper measures to each shot, in place."""
    ts = [f.ts for f in frames]
    times = sorted(s["t"] for s in shots)
    for s in shots:
        i = bisect.bisect_left(ts, s["t"])
        if i >= len(frames):
            s["outcome"], s["speed_1m"] = "open", None
            continue
        f0 = frames[i]
        side = s["side"]
        g = _goal_sign(f0, side)
        goal_x = g * HALF_LEN
        ox, oy = s["raw_x"], s["raw_y"]
        aim_y = s["y_at_goal"] * g  # back to field coordinates
        kid, kp = keeper(f0, other(side))
        if kp is not None and on_field(kp) and abs(kp.p.x - goal_x) < 2.0:
            s["keeper_line_m"] = round(_seg_dist(kp.p.x, kp.p.y, ox, oy, goal_x, aim_y), 3)
            s["keeper_depth_m"] = round(abs(goal_x - kp.p.x), 3)
            s["keeper_y_off"] = round(abs(kp.p.y - aim_y), 3)
        nxt = next((t for t in times if t > s["t"] + 1e-6), math.inf)
        outcome, speed_1m, cross_y, prev = "open", None, None, None
        recent = []  # (ts, vx, vy) over the last POST_LOOKBACK_S, for a bounce off a post
        for f in frames[i:]:
            dt = f.ts - s["t"]
            if dt > FOLLOW_S or f.ts > nxt + 1e-3:  # the next kick's own frame still counts (a bounce off a post)
                break
            b = f.ball
            if not _seen(b):
                continue
            if speed_1m is None and abs(goal_x - b.p.x) <= 1.0:
                speed_1m = math.hypot(b.v.x, b.v.y)
            crossed = (b.p.x - goal_x) * g >= 0
            if crossed and cross_y is None:
                cross_y = b.p.y
                if prev is not None and b.p.x != prev[0]:  # interpolate to the goal line
                    cross_y = prev[1] + (b.p.y - prev[1]) * (goal_x - prev[0]) / (b.p.x - prev[0])
                cross_y *= g
            prev = (b.p.x, b.p.y)
            if s["scored"]:
                outcome = "goal"
                if speed_1m is not None and cross_y is not None:
                    break
                continue
            if crossed:
                off = abs(cross_y) - HALF_GOAL
                outcome = "post" if abs(off) <= POST_BAND_M else ("wide" if off > 0 else "crossed_no_goal")
                if outcome == "wide":
                    s["miss_m"] = round(off, 3)
                break
            if dt > SKIP_S:
                hit = None
                for side_ in (other(side), side):
                    for rid, r in team(f, side_).items():
                        if on_field(r) and math.hypot(r.p.x - b.p.x, r.p.y - b.p.y) <= TOUCH_M:
                            hit = ("keeper" if rid == kid else "defender") if side_ != side else "teammate"
                            break
                    if hit:
                        break
                if hit:
                    outcome = hit
                    break
                near_post = min(math.hypot(b.p.x - goal_x, b.p.y - y) for y in (HALF_GOAL, -HALF_GOAL))
                if near_post <= POST_NEAR_M and recent and _deflected(recent[0][1:], (b.v.x, b.v.y)):
                    outcome = "post"
                    break
                if math.hypot(b.v.x, b.v.y) < STOP_MPS:
                    outcome = "stopped"
                    break
            recent.append((f.ts, b.v.x, b.v.y))
            while recent and f.ts - recent[0][0] > POST_LOOKBACK_S:
                recent.pop(0)
        s["outcome"] = outcome
        s["speed_1m"] = None if speed_1m is None else round(speed_1m, 2)
        s["cross_y"] = None if cross_y is None else round(cross_y, 3)
