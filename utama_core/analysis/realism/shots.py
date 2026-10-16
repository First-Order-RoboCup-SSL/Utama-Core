"""Kicks, shots and which shot scored, from a replay: our sim's or a converted real log.

One shot definition for both sources, made for comparing them (not `MatchStats`' shot counter,
which only fires while a >= 3.5 m/s ball is already in the attacking third and so misses real
teams' long-range goals):

- A **kick** is the ball's velocity changing by `KICK_DV_MPS` within `KICK_WINDOW_S` to a speed
  of at least `KICK_MPS`, so a kick from a dribble or a one-touch redirect counts.
- A kick is a **shot** if, within `SETTLE_S` of it, in live play, its straight line reaches the
  goal mouth (plus `ON_TARGET_MARGIN_M`) and its speed could carry it there at `REACH_DECEL`.
  Kicks from anywhere count (real teams score from their own half); the side is the one
  attacking that goal.

This is the raw detector. The scorecard and the xG fit drop passes and clearances on top of it
(`shot_outcomes.is_attempt`).

Which shot scored (`link_goals`):

- **Real logs:** the game controller halts on the autoref's POSSIBLE_GOAL, and a human accepts
  it ("Accepted by GC operator") seconds to minutes later, so the score rises late. Each score
  rise is matched to its GOAL game event. The event's `kick_location`, the autoref's own kick
  detection, picks the detected shot whose origin is within `KICK_MATCH_M`, kicked at most
  `GOAL_AFTER_SHOT_S` before the ball last entered that goal. POSSIBLE_GOAL can be entered by
  hand long after the ball was reset, so that entry is searched up to `MANUAL_GOAL_S` back. If
  no shot is there (the autoref's kick is a later tap-in or deflection), the side's latest shot
  in that window scores, flagged `kick_mismatch`. Goals with no shot at all are `missed`, with
  the autoref's kick location, so the detector itself can be checked.
- **Our sim** (no game events): the latest shot of that side at most `GOAL_AFTER_SHOT_S` before
  the ball last entered that goal; our referee awards at once.

Shot features, from the shooter's view (attacking +x) where the kick started: distance, angle,
open goal share against all opponents / the keeper alone / field players alone, keeper
position, peak speed in the first `PEAK_WINDOW_S`, robots on the field, the restart it came
from.
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.analysis.chances import LIVE, open_goal
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.referee.referee_command import RefereeCommand as RC
from utama_core.entities.referee.stage import Stage

HALF_LEN = STANDARD_FIELD_DIMS.full_field_half_length
HALF_GOAL = STANDARD_FIELD_DIMS.half_goal_width
GOAL_DEPTH = STANDARD_FIELD_DIMS.goal_depth
KICK_MPS = 2.0  # ball speed after a kick
REACH_DECEL = 0.4  # m/s^2: a shot must be able to roll to the goal line at this deceleration
KICK_DV_MPS = 1.5  # velocity change within KICK_WINDOW_S that makes a kick (or redirect)
KICK_WINDOW_S = 0.2  # the real tracker smooths velocity over several frames
REFRACTORY_S = 0.25
SETTLE_S = 0.1  # the ball's direction settles over a kick's first frames
ON_TARGET_MARGIN_M = 0.15
PEAK_WINDOW_S = 0.3
GOAL_AFTER_SHOT_S = 5.0  # a goal belongs to a shot this recent (rebounds and deflections included)
KICK_MATCH_M = 0.5  # detected kick origin vs the autoref's kick_location
MANUAL_GOAL_S = 180.0  # how far back a hand-entered POSSIBLE_GOAL can be from the ball crossing
SIM_AWARD_LAG_S = 1.0
SET_PIECE_WINDOW_S = 10.0
POSSESSION_M = 0.15  # nearest robot this close to the ball holds it (ends a restart's spell)
RESTARTS = {
    RC.DIRECT_FREE_YELLOW: ("free_kick", True),
    RC.DIRECT_FREE_BLUE: ("free_kick", False),
    RC.PREPARE_PENALTY_YELLOW: ("penalty", True),
    RC.PREPARE_PENALTY_BLUE: ("penalty", False),
    RC.PREPARE_KICKOFF_YELLOW: ("kickoff", True),
    RC.PREPARE_KICKOFF_BLUE: ("kickoff", False),
}
SHOOTOUT = {Stage.PENALTY_SHOOTOUT, Stage.PENALTY_SHOOTOUT_BREAK}


def team(frame, side):
    return frame.friendly_robots if side == "friendly" else frame.enemy_robots


def other(side):
    return "enemy" if side == "friendly" else "friendly"


def on_field(r):
    return not (math.isnan(r.p.x) or math.isnan(r.p.y))


def attacker_of(frame, goal_sign):
    """The side attacking the goal at x = goal_sign * HALF_LEN."""
    friendly_goal_sign = -1.0 if frame.my_team_is_right else 1.0  # the goal friendly attacks
    return "friendly" if goal_sign == friendly_goal_sign else "enemy"


def keeper(frame, side):
    """(robot id, robot) of `side`'s keeper as the referee names it; (None, None) without a referee."""
    if frame.referee is None:
        return None, None
    yellow = (side == "friendly") == frame.my_team_is_yellow
    kid = (frame.referee.yellow_team if yellow else frame.referee.blue_team).goalkeeper
    return kid, team(frame, side).get(kid)


def _holder(frame):
    b = frame.ball
    best = None
    for side in ("friendly", "enemy"):
        for r in team(frame, side).values():
            if on_field(r):
                d = math.hypot(r.p.x - b.p.x, r.p.y - b.p.y)
                if d <= POSSESSION_M and (best is None or d < best[0]):
                    best = (d, side)
    return None if best is None else best[1]


def _shot(frame, start, side, gsign, ts, restart, y_at):
    defending = other(side)
    goal_x = gsign * HALF_LEN
    kid, kp = keeper(start, defending)
    keeper_ok = kp is not None and on_field(kp) and abs(kp.p.x - goal_x) < 2.0
    field = [(r.p.x, r.p.y) for rid, r in team(start, defending).items() if rid != kid and on_field(r)]
    kxy = [(kp.p.x, kp.p.y)] if keeper_ok else []
    ox, oy = start.ball.p.x, start.ball.p.y
    along = abs(goal_x - ox)
    set_piece = (
        restart is not None
        and restart["side"] == side
        and not restart["lost"]
        and ts - restart["t"] <= SET_PIECE_WINDOW_S
    )
    return {
        "side": side,
        "t": round(ts, 3),
        "raw_x": round(ox, 3),
        "raw_y": round(oy, 3),
        "distance_m": round(math.hypot(along, oy), 3),
        "angle_deg": round(math.degrees(math.atan2(abs(oy), along)), 1),
        "origin_x": round(ox * gsign, 3),
        "origin_y": round(oy * gsign, 3),
        "open_goal": round(open_goal((ox, oy), goal_x, field + kxy), 3),
        "og_keeper": round(open_goal((ox, oy), goal_x, kxy), 3),
        "og_field": round(open_goal((ox, oy), goal_x, field), 3),
        "keeper_in_place": keeper_ok,
        "keeper_x": round(kp.p.x * gsign, 3) if keeper_ok else None,
        "keeper_y": round(kp.p.y * gsign, 3) if keeper_ok else None,
        "peak": math.hypot(frame.ball.v.x, frame.ball.v.y),
        "n_att": sum(on_field(r) for r in team(start, side).values()),
        "n_def": sum(on_field(r) for r in team(start, defending).values()),
        "context": restart["kind"] if set_piece else "open_play",
        "y_at_goal": round(y_at * gsign, 3),
    }


def _goal_events(ref, seen, ts, out):
    """Record each new POSSIBLE_GOAL / GOAL game event once (they are resent every message)."""
    for e in ref.game_events or []:
        kind = e.get("type")
        if kind not in ("POSSIBLE_GOAL", "GOAL"):
            continue
        body = e.get("possible_goal") or e.get("goal") or {}
        key = e.get("id") or (kind, e.get("created_timestamp"))
        if key in seen:
            continue
        seen.add(key)
        loc, kick = body.get("location") or {}, body.get("kick_location")
        out.append(
            {
                "type": kind,
                "t": ts,
                "goal_sign": 1.0 if loc.get("x", 0.0) > 0 else -1.0,
                "kick": None if kick is None else (kick.get("x", 0.0), kick.get("y", 0.0)),
                "message": body.get("message"),
                "origin": e.get("origin"),
            }
        )


def scan(frames) -> dict:
    """One pass over a match's frames: shots, score rises, ball entries into a goal, goal game
    events, live seconds, mean robots on the field per side, and restarts that went live."""
    shots, score_events, in_goal_events, game_events, seen_events = [], [], [], [], set()
    goals = {"friendly": 0, "enemy": 0}
    live_s, robots_live, restarts = 0.0, {"friendly": 0.0, "enemy": 0.0}, 0
    prev_ts, prev_score = None, None
    history, last_kick_t, open_kick, pending = [], -math.inf, None, None
    pending_restart, restart = None, None
    was_in_goal = False
    for frame in frames:
        ref = frame.referee
        cmd = ref.referee_command if ref else None
        stage = ref.stage if ref else None
        ts = frame.ts
        live = cmd in LIVE and stage not in SHOOTOUT
        if prev_ts is not None and 0 < ts - prev_ts < 1.0 and live:
            dt = ts - prev_ts
            live_s += dt
            for s in robots_live:
                robots_live[s] += dt * sum(on_field(r) for r in team(frame, s).values())
        prev_ts = ts
        if ref is not None:
            if stage not in SHOOTOUT:
                _goal_events(ref, seen_events, ts, game_events)
            y, b = ref.yellow_team.score, ref.blue_team.score
            score = {"friendly": y, "enemy": b} if frame.my_team_is_yellow else {"friendly": b, "enemy": y}
            if prev_score is not None and stage not in SHOOTOUT:
                for s in goals:
                    for _ in range(max(0, score[s] - prev_score[s])):
                        goals[s] += 1
                        score_events.append((ts, s, frame.my_team_is_right))
            prev_score = score
        if cmd in RESTARTS:
            kind, yellow = RESTARTS[cmd]
            pending_restart = (kind, "friendly" if yellow == frame.my_team_is_yellow else "enemy")
        elif cmd == RC.NORMAL_START and pending_restart is not None:
            restart = {"kind": pending_restart[0], "side": pending_restart[1], "t": ts, "lost": False}
            pending_restart = None
            restarts += 1
        ball = frame.ball
        if ball is None:
            continue
        if restart is not None and not restart["lost"]:
            h = _holder(frame)
            if h is not None and h != restart["side"]:
                restart["lost"] = True
        in_goal = HALF_LEN < abs(ball.p.x) < HALF_LEN + GOAL_DEPTH + 0.1 and abs(ball.p.y) < HALF_GOAL + 0.05
        if in_goal and not was_in_goal and live:
            in_goal_events.append((ts, attacker_of(frame, 1.0 if ball.p.x > 0 else -1.0)))
        was_in_goal = in_goal
        speed = math.hypot(ball.v.x, ball.v.y)
        if open_kick is not None and ts - open_kick["t"] <= PEAK_WINDOW_S:
            open_kick["peak"] = max(open_kick["peak"], speed)
        history.append(frame)
        while history and ts - history[0].ts > KICK_WINDOW_S:
            history.pop(0)
        before = history[0]
        dv = math.hypot(ball.v.x - before.ball.v.x, ball.v.y - before.ball.v.y)
        if speed >= KICK_MPS and dv >= KICK_DV_MPS and ts - last_kick_t >= REFRACTORY_S:
            last_kick_t = ts
            pending = {"t": ts, "before": before}
        if pending is not None and ts - pending["t"] <= SETTLE_S and speed >= KICK_MPS and ball.v.x != 0:
            gsign = 1.0 if ball.v.x > 0 else -1.0
            t_line = (gsign * HALF_LEN - ball.p.x) / ball.v.x
            y_at = ball.p.y + ball.v.y * t_line
            reaches = speed * speed / (2 * REACH_DECEL) >= speed * t_line
            if live and reaches and abs(y_at) <= HALF_GOAL + ON_TARGET_MARGIN_M:
                start = pending["before"]
                shots.append(_shot(frame, start, attacker_of(frame, gsign), gsign, pending["t"], restart, y_at))
                open_kick, pending = shots[-1], None
    for s in shots:
        s["scored"] = False
        s["peak"] = round(s["peak"], 2)
    return {
        "shots": shots,
        "score_events": score_events,
        "in_goal_events": in_goal_events,
        "game_events": game_events,
        "goals": goals,
        "live_s": live_s,
        "robots_live": robots_live,
        "restarts": restarts,
    }


def _link_real(t_score, side, my_right, accepted, used, game_events, in_goal_events, shots):
    """(shot or None, missed record or None, whether a ball entry was found); None for the
    first two and None for the third when no GOAL event matches the score rise."""
    goal_sign = 1.0 if (side == "friendly") != my_right else -1.0  # the goal `side` attacks
    ev = next(
        (
            i
            for i, e in enumerate(accepted)
            if i not in used and e["goal_sign"] == goal_sign and abs(e["t"] - t_score) <= 1.0
        ),
        None,
    )
    if ev is None:
        return None, None, None
    used.add(ev)
    e = accepted[ev]
    possible = [p for p in game_events if p["type"] == "POSSIBLE_GOAL" and p["kick"] == e["kick"]]
    t_event = possible[0]["t"] if possible else e["t"]
    # The ball's last entry into that goal before the event: a goal the referee entered by
    # hand (origin "UI") comes after the ball was already reset.
    entries = [g[0] for g in in_goal_events if g[1] == side and t_event - MANUAL_GOAL_S <= g[0] <= t_event + 0.5]
    t_cross = entries[-1] if entries else t_event
    cands = [s for s in shots if s["side"] == side and not s["scored"] and 0 <= t_cross - s["t"] <= GOAL_AFTER_SHOT_S]
    near = cands
    if e["kick"] is not None:
        kx, ky = e["kick"]
        near = [s for s in cands if math.hypot(s["raw_x"] - kx, s["raw_y"] - ky) <= KICK_MATCH_M]
    shot = None
    if near:
        shot = near[-1]
    elif cands:  # the autoref's kick is a later touch (tap-in, deflection) we did not count as a shot
        shot = cands[-1]
        shot["kick_mismatch"] = True
    missed = None
    if shot is None:
        missed = {
            "side": side,
            "t_cross": round(t_cross, 2),
            "kick": e["kick"],
            "entry_found": bool(entries),
            "manual": "UI" in (possible[0]["origin"] or []) if possible else None,
        }
    return shot, missed, bool(entries)


def _link_sim(t_score, side, used, in_goal_events, shots):
    entry = next(
        (
            g
            for g in reversed(in_goal_events)
            if g[1] == side and t_score - SIM_AWARD_LAG_S <= g[0] <= t_score + 0.5 and g not in used
        ),
        None,
    )
    if entry is None:
        return None
    used.add(entry)
    return next(
        (
            s
            for s in reversed(shots)
            if s["side"] == side and not s["scored"] and 0 <= entry[0] - s["t"] <= GOAL_AFTER_SHOT_S
        ),
        None,
    )


def link_goals(scanned: dict) -> dict:
    """Mark each scored shot in `scanned["shots"]` (in place) and return how the goals linked:
    `link_source`, `unshot_goals` per side, `missed` (real goals with no detected shot),
    `link_stats` and the GOAL events' confirmation lags."""
    shots, game_events, in_goal_events = scanned["shots"], scanned["game_events"], scanned["in_goal_events"]
    unlinked = {"friendly": 0, "enemy": 0}
    missed: list[dict] = []
    accepted = [e for e in game_events if e["type"] == "GOAL"]
    used: set = set()
    link_stats = {"entry_found": 0, "no_entry": 0}
    for t_score, side, my_right in scanned["score_events"]:
        shot: Optional[dict]
        if accepted:
            shot, miss, entry_found = _link_real(
                t_score, side, my_right, accepted, used, game_events, in_goal_events, shots
            )
            if entry_found is not None:
                link_stats["entry_found" if entry_found else "no_entry"] += 1
            if miss is not None:
                missed.append(miss)
        else:
            shot = _link_sim(t_score, side, used, in_goal_events, shots)
        if shot is not None:
            shot["scored"] = True
        else:
            unlinked[side] += 1
    lags = [
        round(g["t"] - p["t"], 1)
        for g in accepted
        for p in game_events
        if p["type"] == "POSSIBLE_GOAL" and p["kick"] == g["kick"] and p["kick"] is not None
    ]
    return {
        "link_source": "game_events" if accepted else "ball_in_goal",
        "unshot_goals": unlinked,
        "missed": missed,
        "link_stats": link_stats,
        "confirm_lags": lags,
        "goal_messages": [g["message"] for g in accepted],
    }
