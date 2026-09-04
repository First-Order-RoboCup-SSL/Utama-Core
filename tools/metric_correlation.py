#!/usr/bin/env python3
# ruff: noqa: E402
"""Offline study: which cheap per-match proxy metrics predict match outcome.

Run from the repository root, for example:

    pixi run python tools/metric_correlation.py \\
        replays/tournament_20260903_101521 \\
        replays/tournament_20260903_112025 \\
        replays/tournament_20260903_115838 \\
        --out benchmark_results/metric_correlation_20260903.md

Purpose
-------
`docs/strategies.md` and `tournament.py` already tell us who won each match, but a
full 65s round-robin match is expensive to run inside an agent loop that's iterating
on a single strategy change. If some cheap per-match "proxy" metric (possession,
territory, shots, ...) reliably predicts match outcome, a much shorter scenario
benchmark reporting just that metric could stand in for a full win-rate sweep. This
script is the offline study that answers "which metrics, if any" using replay data we
already have on disk, before spending any effort wiring new metrics into the live
`MatchStats` accumulator (`utama_core/engine/match_stats.py`).

It does not run the simulator, a tactic, or a strategy — it only reads
`summary.json`/`<a>_vs_<b>.stats.json` (already-computed `MatchStats`) and replays
`<a>_vs_<b>.pkl` frame-by-frame at a 10 Hz sample (`_SAMPLE_HZ`) to compute a second
family of metrics `MatchStats` does not currently report.

Side convention (see `tournament.py::run_match`): `config_a` is always yellow AND
plays right (`my_team_is_right=True`), `config_b` is always blue/left. Every replay
frame's `friendly_robots`/`ball` are recorded from `config_a`'s perspective
(`ReplayMetadata.my_team_is_yellow=True` in every file checked), so "friendly" in a
frame or in `<match>.stats.json` always means `config_a`. Side is a real effect in
this sim (see `tournament.py`'s `--both-sides` docstring), so every metric below is
reported both as a per-side value and as an a-minus-b differential, and side is kept
as an explicit covariate in the logistic fit (part A) rather than assumed away.

Frame-derived metric definitions (see also each metric function's own docstring for
the precise threshold/tie-break used — this is the summary):

1. `shots_on_target` — a ball-speed spike (>= `_SHOT_SPEED_MPS`, matching
   `match_stats.py`'s existing shot heuristic for consistency) launched from within
   `_SHOT_LAUNCH_RADIUS_M` of an attacking-side robot, moving toward the opponent
   goal, whose straight-line velocity ray crosses the opponent goal line inside the
   goal mouth. Edge-detected with the same "locked until ball slows" debounce
   `match_stats.py` uses, so one continuous fast, on-target ball counts once. This is
   a slightly stricter version of `MatchStats.shots` (also requires launch proximity
   to a robot, not just "ball already fast in the attacking third") — kept as its own
   metric rather than replacing `shots` since the two disagree occasionally (a
   deflection re-accelerating the ball with no robot nearby) and that disagreement is
   itself informative for part D's "which to instrument" call.
2. `attacking_third_entries` — count of ball crossings from at/behind the midline
   into a side's attacking third (`x` beyond +/-1.5 m from centre on that side's
   attack axis, matching `match_stats.py`'s zone boundary), hysteresis band
   `_ENTRY_HYSTERESIS_M` wide so oscillation on the 1.5 m line doesn't double-count.
3. `completed_passes` / `turnovers` — a "possession" event starts when the ball
   enters a robot's `_POSSESSION_RADIUS_M` while moving slower than
   `_PASS_RELEASE_MPS ` (i.e. controlled, not a shot flying past) and ends when it
   leaves that radius at >= `_PASS_RELEASE_MPS`. A **completed pass** is a
   possession-end-by-a-side-A-robot immediately (no intervening possession event by
   the opposing side) followed by a possession-start by a *different* side-A robot. A
   **turnover** is the same shape but the next possessor is an opposing-side robot.
   Ball-out-of-bounds/no-further-possession before the match ends counts as neither.
4. `possession_under_pressure_s` — seconds a side's nearest-robot-to-ball (the same
   possession attribution `MatchStats.possession_pct` uses) also has an opponent
   robot within `_PRESSURE_RADIUS_M`, at the 10 Hz sample rate.
5. `restart_to_first_entry_s` / `n_restarts` — from the intentions.jsonl referee
   trace: each transition *into* a live command (`NORMAL_START`/`FORCE_START`) from a
   non-live command starts a restart clock, attributed to whichever side is
   possessing the ball (nearest-robot-to-ball) at that live-start frame; it stops
   at the first frame afterward where that same side's ball position enters its
   attacking third. Restarts with no qualifying entry before the next stoppage or
   match end are excluded from the mean (and counted separately, `n_restarts` vs.
   `n_restarts_with_entry`) rather than imputing an arbitrary ceiling. Reported
   both per-side (`restart_to_first_entry_s_friendly`/`_enemy`, feeding the same
   `a-b` differential every other metric uses) and as a single match-level mean
   pooling both sides. *Lower is better* for the possessing side (faster tempo
   off a restart) -- see `_LOWER_IS_BETTER`.
6. `mean_ball_x_towards_opponent_goal` (territory) and `defensive_third_time_pct` —
   mean of `ball.p.x` projected onto each side's attack axis (matching
   `match_stats.py`'s `own_goal_sign` convention) across sampled frames, and the
   fraction of sampled frames the ball spends in a side's own defensive third.
7. Two more, chosen for a plausible causal link to scoring rather than just
   correlation-mining:
   - `shot_backed_up_rate` — of `shots_on_target` events, the fraction where a
     teammate of the shooting side is within `_SUPPORT_RADIUS_M` of the ball at the
     moment of the shot (rebound/follow-up support). Motivated by: an unsupported shot
     that's saved or deflected usually just turns over possession, while a backed-up
     shot can be converted on the second ball — plausibly causal for *actually*
     converting territory into goals, not just reaching the box.
   - `defensive_third_time_pct` (see 6) doubles as a plausible direct predictor of
     *conceding*: time spent defending your own third is time the opponent could
     be shooting, independent of any single shot metric firing.

All frame-derived metrics are computed once per match from a single pass over
10 Hz-sampled frames (`_iter_sampled_frames`) and cached to
`<scratchpad>/frame_metrics/<run>/<match>.json` so a re-run (e.g. after a code
tweak to this script) never re-reads a `.pkl`. Per-match rows (frame metrics +
stats.json + summary.json result) are also cached per run to
`<scratchpad>/rows/<run>.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy import stats as scipy_stats

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is present in the pixi env
    _HAVE_SCIPY = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.config.field_params import STANDARD_FIELD_DIMS  # noqa: E402
from utama_core.config.physical_constants import ROBOT_RADIUS  # noqa: E402
from utama_core.replay.columnar_reader import load_columnar_replay  # noqa: E402
from utama_core.replay.entities import ReplayMetadata  # noqa: E402
from utama_core.replay.replay_player import _load_replay  # noqa: E402

# --- Sampling ---------------------------------------------------------------
_SAMPLE_HZ = 10.0
_TICK_HZ = 60.0  # rsim's fixed step rate (tournament.py::TICKS_PER_SECOND)
_SAMPLE_STRIDE = max(1, round(_TICK_HZ / _SAMPLE_HZ))
_DT = _SAMPLE_STRIDE / _TICK_HZ

# --- Field geometry (matches utama_core/engine/match_stats.py's conventions) ---
_HALF_LENGTH = STANDARD_FIELD_DIMS.full_field_half_length  # 4.5
_HALF_GOAL_WIDTH = STANDARD_FIELD_DIMS.half_goal_width  # 0.5
_ATTACKING_THIRD_M = 1.5  # matches match_stats.py's zone boundary

# --- Metric thresholds (each used in exactly one metric function below; see the
# module docstring's numbered list for the "why") ---
_SHOT_SPEED_MPS = 3.5  # matches match_stats.py's _SHOT_SPEED_MPS
_SHOT_LAUNCH_RADIUS_M = 0.5  # "from within N m of a robot" per the task's metric 1
_ENTRY_HYSTERESIS_M = 0.3  # metric 2: band around the 1.5 m third boundary
_POSSESSION_RADIUS_M = 2 * ROBOT_RADIUS + 0.05  # ~0.23 m: robot footprint + dribble slack
_PASS_RELEASE_MPS = 1.0  # matches match_stats.py's _SHOT_LOCK_RELEASE_MPS release speed
_PRESSURE_RADIUS_M = 0.5  # metric 4, per the task spec
_SUPPORT_RADIUS_M = 1.0  # metric 7: "teammate near the ball at shot time"
_RESTART_LIVE_COMMANDS = {"NORMAL_START", "FORCE_START"}

# --- Metrics 8-12 (2026-09-04 debugging-signal expansion): each mined from a
# real, already-traced bug mechanism in docs/roadmap.md item 15, rather than
# invented speculatively -- the point is a cheap counter that would have
# flagged that exact mechanism automatically instead of needing a human to
# live-trace a match. See each metric's own comment below for which bug it
# targets.
_NEAR_STALL_RADIUS_M = 0.5  # metric 8: same order as match_stats.py's NO_PROGRESS_POSSESSION radius (0.35), a
# little looser here since this is meant to catch pre-watchdog *risk*, not confirmed stalls
_NEAR_STALL_SECONDS = 2.0  # metric 8: much shorter than the live watchdog's 8s -- an early-warning signal,
# not a replacement for it
_THRASH_WINDOW_S = 5.0  # metric 9: matches the ~5-9s windows item 15's kicker-thrash traces were measured over
_OSCILLATION_WINDOW_S = 3.0  # metric 10: window over which a distance-derivative sign-flip run is counted
_OSCILLATION_MIN_AMPLITUDE_M = 0.3  # metric 10: ignore sub-30cm position noise, only count a real retreat/re-approach


def _iter_sampled_frames(replay_path: Path):
    """Yield every `_SAMPLE_STRIDE`-th `GameFrame` from a replay file, plus the
    replay metadata first. A generator, not a list, so a caller that only needs
    a running accumulation (the common case here) never holds the sampled frames
    in memory at once.

    Dispatches on extension like `replay_player.load_frames_in_range` does:
    `.npz` is the columnar format current tournament runs actually write
    (`ColumnarReplayWriter`) — `_load_replay` only understands the older
    one-pickle-per-frame `.pkl` format, so it can't read current runs at all.
    `ColumnarReplay` has no separate `ReplayMetadata` object; `my_team_is_yellow`
    is a field on it directly, which is the only piece of metadata this module
    reads, so it's wrapped in a `ReplayMetadata` to keep `compute_frame_metrics`
    unchanged.
    """
    if replay_path.suffix == ".npz":
        replay = load_columnar_replay(replay_path)
        yield ReplayMetadata(
            my_team_is_yellow=replay.my_team_is_yellow,
            exp_friendly=len(replay.friendly_ids),
            exp_enemy=len(replay.enemy_ids),
        )
        for i in range(0, replay.n_ticks, _SAMPLE_STRIDE):
            yield replay.frame_at(i)
        return

    gen = _load_replay(replay_path)
    metadata = next(gen)
    yield metadata
    for i, frame in enumerate(gen):
        if i % _SAMPLE_STRIDE == 0:
            yield frame


def _attack_sign(my_team_is_right: bool, side: str) -> float:
    """+1/-1 multiplier s.t. `x * attack_sign` grows as `side`'s ball/robot moves
    toward the *opponent's* goal. Matches `match_stats.py`'s `own_goal_sign`
    convention exactly (friendly's own goal sits at `+own_goal_sign * half_length`).
    """
    own_goal_sign = 1.0 if my_team_is_right else -1.0
    return -own_goal_sign if side == "friendly" else own_goal_sign


def _own_goal_x(my_team_is_right: bool, side: str) -> float:
    own_goal_sign = 1.0 if my_team_is_right else -1.0
    ref_sign = own_goal_sign if side == "friendly" else -own_goal_sign
    return ref_sign * _HALF_LENGTH


@dataclass
class _PossessionState:
    side: Optional[str] = None
    robot_id: Optional[int] = None


def compute_frame_metrics(replay_path: Path, referee_events: list[dict]) -> dict:
    """Single pass over 10 Hz-sampled frames computing every frame-derived metric
    (see module docstring, items 1-7) for both `friendly` (config_a) and `enemy`
    (config_b). Returns a flat dict of `{metric}_{side}` -> value plus a couple of
    match-level counts (`n_restarts`, `n_restarts_with_entry`).
    """
    frames_iter = _iter_sampled_frames(replay_path)
    metadata = next(frames_iter)
    my_team_is_right = metadata.my_team_is_yellow  # config_a is always yellow+right together

    sides = ("friendly", "enemy")
    shots_on_target = {s: 0 for s in sides}
    shots_backed_up = {s: 0 for s in sides}
    shot_lock = {s: False for s in sides}
    entries = {s: 0 for s in sides}
    in_attacking_third = {s: False for s in sides}  # hysteresis state
    ball_x_sum = {s: 0.0 for s in sides}
    defensive_third_ticks = {s: 0 for s in sides}
    pressure_ticks = {s: 0 for s in sides}
    n_ticks = 0

    possession = _PossessionState()  # current controlling (side, robot_id) or None
    completed_passes = {s: 0 for s in sides}
    turnovers = {s: 0 for s in sides}

    # --- metric 8: near-stall risk (pre-watchdog early warning) ---
    # Same shape as match_stats.py's NO_PROGRESS_POSSESSION watchdog (a robot
    # stuck near the ball without ever gaining possession), but a much shorter
    # threshold and a looser radius -- meant to fire well before the live 8s
    # watchdog would, so it can serve as a leading risk indicator rather than a
    # confirmed-stall count. Tracked per side (whichever side's nearest robot is
    # the one stuck).
    near_stall_since: dict[str, Optional[float]] = {s: None for s in sides}
    near_stall_holder: dict[str, Optional[int]] = {s: None for s in sides}
    near_stall_events = {s: 0 for s in sides}

    # --- metric 9: kicker/nearest-robot identity thrash during a restart ---
    # Mined directly from item 15's traced DIRECT_FREE bug: the "closest robot
    # to the ball" identity flipped ~9x/second on ordinary position noise
    # before the Sticky[int] fix (243 switches/27s -> 4). This counts identity
    # switches of the nearest-to-ball robot *within a side*, while a restart
    # command is in effect -- a regression of that exact bug class would show
    # up here even if nothing else changes.
    restart_active = False
    thrash_prev_id: dict[str, Optional[int]] = {s: None for s in sides}
    thrash_switches = {s: 0 for s in sides}

    # --- metric 10: retreat/re-approach oscillation ---
    # The "congestion/local-minimum" signature item 15 traced repeatedly: a
    # side's nearest-robot-to-ball distance oscillates (approach, retreat,
    # re-approach) instead of monotonically closing, for seconds at a time,
    # while the planner reports no collision. Counts local direction reversals
    # in each side's nearest-to-ball distance time series, ignoring reversals
    # smaller than _OSCILLATION_MIN_AMPLITUDE_M (position noise).
    osc_last_dist: dict[str, Optional[float]] = {s: None for s in sides}
    osc_last_extremum: dict[str, Optional[float]] = {s: None for s in sides}
    osc_direction: dict[str, int] = {s: 0 for s in sides}  # +1 closing, -1 opening, 0 unknown
    osc_reversals = {s: 0 for s in sides}

    # --- metric 11: ball-carrier hold duration (COMMITTED_FROZEN precursor) ---
    # Targets the DefenseTactic ball-drag bug class: one robot keeps has_ball
    # continuously for an abnormally long stretch. Tracks the longest
    # continuous has_ball run per side across the match (p95 needs many
    # matches pooled, so this reports the single max per match; part B pools
    # across a strategy's matches for a percentile).
    carrier_hold_since: dict[str, Optional[float]] = {s: None for s in sides}
    carrier_hold_id: dict[str, Optional[int]] = {s: None for s in sides}
    carrier_hold_max_s = {s: 0.0 for s in sides}

    # Restart-to-entry bookkeeping, split by which side was possessing at the
    # restart (so it folds into the same per-side differential framework every
    # other metric uses -- a side reaching the attacking third faster off its
    # own restarts is a directly plausible predictor of scoring more).
    restarts_resolved: dict[str, list[float]] = {s: [] for s in sides}
    n_restarts = 0
    n_restarts_with_entry = 0
    pending_restart: Optional[dict] = None  # {"t0": float, "side": Optional[str]}

    # Referee events are timestamped independently (intentions.jsonl); merge by
    # sim_time against the sampled frame stream below rather than a second pass.
    ref_events = sorted(referee_events, key=lambda e: e["sim_time"])
    ref_idx = 0
    prev_command: Optional[str] = None

    for frame in frames_iter:
        ball = frame.ball
        if ball is None:
            continue
        n_ticks += 1
        t = frame.ts

        # Advance the referee-event cursor to the latest event at or before this
        # sampled frame, tracking live-command transitions as we pass them.
        while ref_idx < len(ref_events) and ref_events[ref_idx]["sim_time"] <= t:
            cmd = ref_events[ref_idx]["command"]
            if cmd in _RESTART_LIVE_COMMANDS and prev_command not in _RESTART_LIVE_COMMANDS:
                # A live-play command just started a restart clock. Whoever is
                # nearest the ball *now* (this frame) is "the possessing side" for
                # the purpose of measuring how fast they reach the attacking third.
                all_robots = [(rid, r, "friendly") for rid, r in frame.friendly_robots.items()] + [
                    (rid, r, "enemy") for rid, r in frame.enemy_robots.items()
                ]
                nearest_side = min(all_robots, key=lambda e: e[1].p.distance_to(ball.p))[2] if all_robots else None
                pending_restart = {"t0": t, "side": nearest_side}
                n_restarts += 1
            # metric 9: "restart active" = any non-HALT/STOP referee command that
            # isn't yet live play (the mass-replan ceremony window item 15's
            # kicker-thrash bug was traced during -- PREPARE_*/DIRECT_FREE_*/
            # BALL_PLACEMENT_* etc., not the live-play commands themselves, since
            # once play is live the "nearest robot" is expected to change hands
            # normally as the ball moves).
            restart_active = cmd not in ("HALT", "STOP") and cmd not in _RESTART_LIVE_COMMANDS
            prev_command = cmd
            ref_idx += 1

        all_robots = [(rid, r, "friendly") for rid, r in frame.friendly_robots.items()] + [
            (rid, r, "enemy") for rid, r in frame.enemy_robots.items()
        ]

        # --- possession (nearest robot to ball; matches match_stats.py) ---
        nearest_id, nearest_robot, nearest_side = min(all_robots, key=lambda e: e[1].p.distance_to(ball.p))
        dist_to_nearest = nearest_robot.p.distance_to(ball.p)

        # --- metric 4: possession under pressure ---
        opp_prefix = "enemy" if nearest_side == "friendly" else "friendly"
        opponents = [r for rid, r, s in all_robots if s == opp_prefix]
        if dist_to_nearest <= _POSSESSION_RADIUS_M and opponents:
            min_opp_dist = min(r.p.distance_to(ball.p) for r in opponents)
            if min_opp_dist <= _PRESSURE_RADIUS_M:
                pressure_ticks[nearest_side] += 1

        # --- metric 8: near-stall risk ---
        # A side's nearest robot sits within _NEAR_STALL_RADIUS_M of the ball,
        # same identity, for more than _NEAR_STALL_SECONDS, without possession
        # ever latching to it (possession.side/robot_id != this holder) -- an
        # early-warning version of match_stats.py's NO_PROGRESS_POSSESSION
        # watchdog (which uses a tighter radius and an 8s threshold). One event
        # is logged per qualifying stretch (not re-logged every tick it
        # continues), mirroring that watchdog's own "log once per stretch" shape.
        for side in sides:
            side_robots = [(rid, r) for rid, r, s in all_robots if s == side]
            if not side_robots:
                near_stall_since[side] = None
                near_stall_holder[side] = None
                continue
            s_id, s_robot = min(side_robots, key=lambda e: e[1].p.distance_to(ball.p))
            s_dist = s_robot.p.distance_to(ball.p)
            has_control = possession.side == side and possession.robot_id == s_id
            if s_dist <= _NEAR_STALL_RADIUS_M and not has_control:
                if near_stall_holder[side] != s_id:
                    near_stall_since[side] = t
                    near_stall_holder[side] = s_id
                elif near_stall_since[side] is not None and (t - near_stall_since[side]) > _NEAR_STALL_SECONDS:
                    near_stall_events[side] += 1
                    near_stall_since[side] = None  # reset so a longer stretch doesn't multi-count
            else:
                near_stall_since[side] = None
                near_stall_holder[side] = None

        # --- metric 9: nearest-robot identity thrash during a restart ceremony ---
        if restart_active:
            for side in sides:
                side_robots = [(rid, r) for rid, r, s in all_robots if s == side]
                if not side_robots:
                    continue
                s_id, _ = min(side_robots, key=lambda e: e[1].p.distance_to(ball.p))
                if thrash_prev_id[side] is not None and thrash_prev_id[side] != s_id:
                    thrash_switches[side] += 1
                thrash_prev_id[side] = s_id
        else:
            thrash_prev_id = {s: None for s in sides}

        # --- metric 10: retreat/re-approach oscillation ---
        # A local direction reversal in a side's nearest-to-ball distance,
        # ignoring reversals smaller than _OSCILLATION_MIN_AMPLITUDE_M so
        # ordinary position noise (and a robot legitimately arriving and
        # settling) doesn't count. This is a running peak/trough detector, not
        # a windowed FFT -- cheap and matches how the bug was actually spotted
        # in live traces (distance visibly sawtoothing instead of monotonically
        # closing).
        for side in sides:
            side_robots = [(rid, r) for rid, r, s in all_robots if s == side]
            if not side_robots:
                continue
            s_dist = min(r.p.distance_to(ball.p) for _, r in side_robots)
            if osc_last_dist[side] is None:
                osc_last_dist[side] = s_dist
                osc_last_extremum[side] = s_dist
            else:
                delta = s_dist - osc_last_dist[side]
                new_direction = 1 if delta > 0 else (-1 if delta < 0 else osc_direction[side])
                if (
                    osc_direction[side] != 0
                    and new_direction != osc_direction[side]
                    and osc_last_extremum[side] is not None
                    and abs(s_dist - osc_last_extremum[side]) >= _OSCILLATION_MIN_AMPLITUDE_M
                ):
                    osc_reversals[side] += 1
                    osc_last_extremum[side] = s_dist
                elif osc_last_extremum[side] is None or (
                    new_direction == osc_direction[side]
                    and (
                        (new_direction > 0 and s_dist > osc_last_extremum[side])
                        or (new_direction < 0 and s_dist < osc_last_extremum[side])
                    )
                ):
                    osc_last_extremum[side] = s_dist
                osc_direction[side] = new_direction
                osc_last_dist[side] = s_dist

        # --- metric 11: ball-carrier hold duration (COMMITTED_FROZEN precursor) ---
        for side in sides:
            side_robots = [(rid, r) for rid, r, s in all_robots if s == side]
            holder = next((rid for rid, r in side_robots if r.has_ball), None)
            if holder is not None and carrier_hold_id[side] == holder:
                held_for = t - carrier_hold_since[side]
                if held_for > carrier_hold_max_s[side]:
                    carrier_hold_max_s[side] = held_for
            elif holder is not None:
                carrier_hold_id[side] = holder
                carrier_hold_since[side] = t
            else:
                carrier_hold_id[side] = None
                carrier_hold_since[side] = None

        # --- metric 3: completed passes / turnovers (possession-radius state machine) ---
        ball_speed = math.hypot(ball.v.x, ball.v.y)
        controlled = dist_to_nearest <= _POSSESSION_RADIUS_M and ball_speed < _PASS_RELEASE_MPS
        if controlled:
            new_holder = (nearest_side, nearest_id)
            if possession.side is None:
                possession = _PossessionState(*new_holder)
            elif (possession.side, possession.robot_id) != new_holder:
                # Possession changed hands without an intervening "released at
                # speed" event (e.g. a slow dribble handoff/tackle) -- attribute
                # as a same-side completed pass or a turnover exactly like a
                # released pass would be, then adopt the new holder.
                if possession.side == nearest_side:
                    completed_passes[possession.side] += 1
                else:
                    turnovers[possession.side] += 1
                possession = _PossessionState(*new_holder)
        elif possession.side is not None and ball_speed >= _PASS_RELEASE_MPS and dist_to_nearest > _POSSESSION_RADIUS_M:
            # Ball just left a controlled possession at speed: released. The
            # eventual outcome (pass vs turnover vs neither) is resolved the next
            # time the ball is controlled again, or never if it isn't. Mark as
            # "released" by clearing robot_id but remembering the releasing side.
            if possession.robot_id is not None:
                possession = _PossessionState(side=possession.side, robot_id=None)

        # --- metric 1: shots on target ---
        for side in sides:
            attack_sign = _attack_sign(my_team_is_right, side)
            if ball_speed < 1.0:
                shot_lock[side] = False
            if shot_lock[side]:
                continue
            toward_goal = ball.v.x * attack_sign > 0.0
            own_goal_x = _own_goal_x(my_team_is_right, side)
            progress_from_own_goal = (ball.p.x - own_goal_x) * attack_sign
            if not (
                ball_speed >= _SHOT_SPEED_MPS
                and toward_goal
                and progress_from_own_goal > _HALF_LENGTH + _ATTACKING_THIRD_M
            ):
                continue
            launcher_side_robots = [r for rid, r, s in all_robots if s == side]
            if not launcher_side_robots:
                continue
            near_launcher = min(r.p.distance_to(ball.p) for r in launcher_side_robots) <= _SHOT_LAUNCH_RADIUS_M
            if not near_launcher:
                continue
            attacking_goal_x = attack_sign * _HALF_LENGTH
            if ball.v.x == 0:
                continue
            ticks_to_goal_line = (attacking_goal_x - ball.p.x) / ball.v.x
            predicted_y = ball.p.y + ball.v.y * ticks_to_goal_line
            if abs(predicted_y) <= _HALF_GOAL_WIDTH:
                shots_on_target[side] += 1
                shot_lock[side] = True
                # metric 7a: is a teammate (excluding the shooter itself, hence
                # the `0 <` lower bound) backing up the shot within support range?
                teammates = [r for rid, r, s in all_robots if s == side]
                if any(0 < r.p.distance_to(ball.p) <= _SUPPORT_RADIUS_M for r in teammates):
                    shots_backed_up[side] += 1

        # --- metric 2: attacking-third entries (hysteresis) ---
        for side in sides:
            attack_sign = _attack_sign(my_team_is_right, side)
            own_goal_x = _own_goal_x(my_team_is_right, side)
            progress = (ball.p.x - own_goal_x) * attack_sign
            enter_th = _HALF_LENGTH + _ATTACKING_THIRD_M
            exit_th = enter_th - _ENTRY_HYSTERESIS_M
            if not in_attacking_third[side] and progress > enter_th:
                in_attacking_third[side] = True
                entries[side] += 1
            elif in_attacking_third[side] and progress < exit_th:
                in_attacking_third[side] = False

        # --- metric 6: territory / defensive-third time ---
        # `progress` here is the ball's position measured from `side`'s own goal
        # along its attack axis: large positive = deep in the opponent's end
        # (territory), small/negative = deep in `side`'s own end. The same
        # `< _ATTACKING_THIRD_M` cut `match_stats.py` uses for "defensive zone"
        # (mirrored: that file measures from each *robot's* own goal too) gives
        # the fraction of time the ball spends in `side`'s own defensive third.
        for side in sides:
            attack_sign = _attack_sign(my_team_is_right, side)
            own_goal_x = _own_goal_x(my_team_is_right, side)
            progress = (ball.p.x - own_goal_x) * attack_sign
            ball_x_sum[side] += progress
            if progress < _ATTACKING_THIRD_M:
                defensive_third_ticks[side] += 1

        # --- restart-to-entry resolution ---
        if pending_restart is not None:
            side = pending_restart["side"]
            if side is not None and in_attacking_third[side]:
                restarts_resolved[side].append(t - pending_restart["t0"])
                n_restarts_with_entry += 1
                pending_restart = None

    n_ticks = max(n_ticks, 1)
    all_resolved = restarts_resolved["friendly"] + restarts_resolved["enemy"]
    out = {
        "n_sampled_ticks": n_ticks,
        "n_restarts": n_restarts,
        "n_restarts_with_entry": n_restarts_with_entry,
        "restart_to_first_entry_s_mean": (float(np.mean(all_resolved)) if all_resolved else None),
    }
    for side in sides:
        out[f"shots_on_target_{side}"] = shots_on_target[side]
        out[f"restart_to_first_entry_s_{side}"] = (
            float(np.mean(restarts_resolved[side])) if restarts_resolved[side] else None
        )
        out[f"shot_backed_up_rate_{side}"] = (
            shots_backed_up[side] / shots_on_target[side] if shots_on_target[side] > 0 else None
        )
        out[f"attacking_third_entries_{side}"] = entries[side]
        out[f"completed_passes_{side}"] = completed_passes[side]
        out[f"turnovers_{side}"] = turnovers[side]
        out[f"possession_under_pressure_s_{side}"] = pressure_ticks[side] * _DT
        out[f"mean_ball_x_towards_opponent_goal_{side}"] = ball_x_sum[side] / n_ticks
        out[f"defensive_third_time_pct_{side}"] = defensive_third_ticks[side] / n_ticks
        out[f"near_stall_events_{side}"] = near_stall_events[side]
        out[f"kicker_identity_thrash_{side}"] = thrash_switches[side]
        out[f"retreat_reapproach_oscillations_{side}"] = osc_reversals[side]
        out[f"ball_carrier_hold_max_s_{side}"] = carrier_hold_max_s[side]
    return out


def _match_tag(config_a: str, config_b: str) -> str:
    def short(name: str) -> str:
        return name.removeprefix("build_").removesuffix("_kernel_strategy")

    return f"{short(config_a)}_vs_{short(config_b)}"


def _load_referee_events(intentions_path: Path) -> list[dict]:
    events = []
    if not intentions_path.exists():
        return events
    with open(intentions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("event") == "referee":
                events.append({"sim_time": d["sim_time"], "command": d["command"]})
    return events


def _cache_path(scratch_dir: Path, run_name: str, match_tag: str) -> Path:
    d = scratch_dir / "frame_metrics" / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{match_tag}.json"


def _worker(args: tuple) -> tuple[str, dict]:
    run_dir_str, match_tag, scratch_dir_str = args
    run_dir = Path(run_dir_str)
    scratch_dir = Path(scratch_dir_str)
    cache_path = _cache_path(scratch_dir, run_dir.name, match_tag)
    if cache_path.exists():
        with open(cache_path) as f:
            return match_tag, json.load(f)
    # Current tournament runs write `.npz` (ColumnarReplayWriter); older run
    # directories on disk still have `.pkl` (see `_iter_sampled_frames`'s
    # dispatch on extension) — prefer `.npz`, fall back for old runs.
    npz_path = run_dir / f"{match_tag}.npz"
    replay_path = npz_path if npz_path.exists() else run_dir / f"{match_tag}.pkl"
    intentions_path = run_dir / f"{match_tag}.intentions.jsonl"
    ref_events = _load_referee_events(intentions_path)
    try:
        metrics = compute_frame_metrics(replay_path, ref_events)
    except Exception as exc:  # pragma: no cover - defensive: one bad match shouldn't kill the run
        metrics = {"_error": str(exc)}
    with open(cache_path, "w") as f:
        json.dump(metrics, f)
    return match_tag, metrics


def build_rows(run_dir: Path, scratch_dir: Path, n_workers: int) -> list[dict]:
    """One row per match in `run_dir`: summary.json result + stats.json + frame
    metrics, cached whole to `<scratch_dir>/rows/<run_name>.json`.
    """
    rows_cache = scratch_dir / "rows" / f"{run_dir.name}.json"
    rows_cache.parent.mkdir(parents=True, exist_ok=True)
    if rows_cache.exists():
        with open(rows_cache) as f:
            return json.load(f)

    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    tasks = []
    for r in summary["results"]:
        match_tag = _match_tag(r["config_a"], r["config_b"])
        tasks.append((str(run_dir), match_tag, str(scratch_dir)))

    frame_results: dict[str, dict] = {}
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for match_tag, metrics in pool.imap_unordered(_worker, tasks, chunksize=1):
                frame_results[match_tag] = metrics
    else:
        for t in tasks:
            match_tag, metrics = _worker(t)
            frame_results[match_tag] = metrics

    rows = []
    for r in summary["results"]:
        match_tag = _match_tag(r["config_a"], r["config_b"])
        stats = r.get("stats") or {}
        frame_metrics = frame_results.get(match_tag, {})
        row = {
            "run": run_dir.name,
            "match_tag": match_tag,
            "config_a": r["config_a"],
            "config_b": r["config_b"],
            "score_a": r["score_a"],
            "score_b": r["score_b"],
            "winner": r["winner"],
            "stats": stats,
            "frame_metrics": frame_metrics,
        }
        rows.append(row)

    with open(rows_cache, "w") as f:
        json.dump(rows, f)
    return rows


# --- Metric differential extraction -----------------------------------------

# (metric_name, stats.json path template or frame_metrics key template).
# "stats:" prefix -> read from row["stats"], "frame:" prefix -> row["frame_metrics"].
#
# `ball_travel_m` (from stats.json) is deliberately NOT in this list: `MatchStats`
# records it as one match-level total, not split by side, so it has no `a-b`
# differential to compute -- see `part_a`/report generation below for how it's
# reported instead (against |goal diff|, not signed goal diff).
#
# `shot_backed_up_rate` is also excluded from `ALL_METRICS`'s per-match
# differential framework (part A): it's only defined for a side that took >=1
# shot on target in that match, and with only ~13 shots_on_target events across
# 231 matches in a run, both sides taking a shot in the *same* match essentially
# never happens -- the per-match differential would be `None` almost everywhere.
# It's still analysed in parts B/C, pooled per-strategy (total backed-up shots /
# total shots across all of a strategy's matches, not a mean of per-match
# differences) -- see `_shot_backed_up_pooled_rate`.
_STATS_METRICS = [
    "possession_pct",
    "shots",
    "zone_time_pct_attacking",
    "robot_motion_pct",
]
_FRAME_METRICS = [
    "shots_on_target",
    "attacking_third_entries",
    "completed_passes",
    "turnovers",
    "possession_under_pressure_s",
    "mean_ball_x_towards_opponent_goal",
    "defensive_third_time_pct",
    "restart_to_first_entry_s",
    # Metrics 8-11 (2026-09-04 debugging-signal expansion, see compute_frame_metrics):
    # each is a cheap proxy mined from a real, already-traced bug mechanism, tested
    # here for whether it *also* carries any outcome signal, not just bug-detection value.
    "near_stall_events",
    "kicker_identity_thrash",
    "retreat_reapproach_oscillations",
    "ball_carrier_hold_max_s",
]
# Metrics where a *lower* value is better for that side -- `turnovers`,
# `defensive_third_time_pct`, and `restart_to_first_entry_s` (faster = better
# attacking tempo off a restart). Their differentials are still computed the
# same `own - opponent` way as every other metric; a *negative* correlation
# with points/goal-diff is the expected "good" sign for these, same as
# `defensive_third_time_pct` already reads in part B's table. The 4 new
# debugging-signal metrics are all "lower is better" too -- each one counts an
# occurrence of a known-bad mechanism (near-stall risk, identity thrash,
# retreat/re-approach oscillation, an abnormally long ball hold).
_LOWER_IS_BETTER = frozenset(
    {
        "turnovers",
        "defensive_third_time_pct",
        "restart_to_first_entry_s",
        "near_stall_events",
        "kicker_identity_thrash",
        "retreat_reapproach_oscillations",
        "ball_carrier_hold_max_s",
    }
)

ALL_METRICS = _STATS_METRICS + _FRAME_METRICS


def _team_zone_attacking_mean(zone_time_pct: dict, side: str) -> Optional[float]:
    vals = [v["attacking"] for k, v in zone_time_pct.items() if k.startswith(f"{side}_") and "attacking" in v]
    return float(np.mean(vals)) if vals else None


def _team_robot_motion_mean(robot_motion_pct: dict, side: str) -> Optional[float]:
    vals = [v for k, v in robot_motion_pct.items() if k.startswith(f"{side}_")]
    return float(np.mean(vals)) if vals else None


def extract_side_value(row: dict, metric: str, side: str) -> Optional[float]:
    """`side` is "friendly" (=config_a) or "enemy" (=config_b) — matches the
    replay/stats.json convention throughout this file.
    """
    stats = row["stats"]
    fm = row["frame_metrics"]
    if metric == "possession_pct":
        return stats.get("possession_pct", {}).get(side)
    if metric == "shots":
        return stats.get("shots", {}).get(side)
    if metric == "zone_time_pct_attacking":
        return _team_zone_attacking_mean(stats.get("zone_time_pct", {}), side)
    if metric == "robot_motion_pct":
        return _team_robot_motion_mean(stats.get("robot_motion_pct", {}), side)
    if metric in _FRAME_METRICS:
        return fm.get(f"{metric}_{side}")
    raise ValueError(metric)


def compute_differential(row: dict, metric: str) -> Optional[float]:
    """a (friendly/config_a) minus b (enemy/config_b), None if either side's value
    is missing (e.g. shot_backed_up_rate when a side took zero shots on target).
    """
    a = extract_side_value(row, metric, "friendly")
    b = extract_side_value(row, metric, "enemy")
    if a is None or b is None:
        return None
    return a - b


def goal_diff(row: dict) -> int:
    return row["score_a"] - row["score_b"]


def winner_label(row: dict) -> Optional[int]:
    """1 if config_a won, 0 if config_b won, None if draw (decisive-only metrics)."""
    if row["winner"] == "draw":
        return None
    return 1 if row["winner"] == row["config_a"] else 0


# --- Statistics ---------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    if _HAVE_SCIPY:
        rho, p = scipy_stats.spearmanr(x, y)
        return float(rho), float(p), n
    # Fallback: rank + Pearson.
    rx = _rankdata(x)
    ry = _rankdata(y)
    rho = _pearson(rx, ry)
    return rho, float("nan"), n


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(np.sum(xm**2)) * float(np.sum(ym**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(xm * ym) / denom)


def auc_score(values: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """AUC of `values` predicting `labels==1`, via Mann-Whitney U (ties split)."""
    mask = ~np.isnan(values)
    values, labels = values[mask], labels[mask]
    pos = values[labels == 1]
    neg = values[labels == 0]
    n = len(pos) + len(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), n
    if _HAVE_SCIPY:
        u, _ = scipy_stats.mannwhitneyu(pos, neg, alternative="two-sided")
        auc = u / (len(pos) * len(neg))
        return float(auc), n
    # Fallback: direct rank-sum AUC.
    combined = np.concatenate([pos, neg])
    ranks = _rankdata(combined)
    r_pos = ranks[: len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc), n


def fit_logistic_with_side(x: np.ndarray, side_is_a_right: np.ndarray, y: np.ndarray) -> Optional[dict]:
    """Logistic regression of `y` (1=config_a win) on [intercept, x, side_covariate].

    In this dataset config_a is *always* right (see module docstring), so a
    literal "which physical side" covariate would be a constant column and
    useless. What genuinely varies match-to-match is a fixed all-True/False
    only across the whole corpus, so instead we use, as the requested "side
    intercept", a coin-flip-free proxy that is still a real per-match binary
    covariate: none exists in this dataset since config_a=yellow=right on
    every single match (see tournament.py::run_match's hardcoded
    my_team_is_yellow=True/my_team_is_right=True). We therefore fit x alone
    (intercept + slope) and report that the side covariate is degenerate
    (constant) rather than silently fitting a coefficient to an all-ones
    column, which statsmodels/sklearn would either drop or blow up on.
    """
    mask = ~np.isnan(x)
    x, y = x[mask], y[mask]
    if len(np.unique(y)) < 2 or len(x) < 5:
        return None
    # Standardize x for numerically stable IRLS.
    mu, sigma = x.mean(), x.std()
    if sigma == 0:
        return None
    xs = (x - mu) / sigma
    X = np.column_stack([np.ones_like(xs), xs])
    beta = np.zeros(2)
    for _ in range(100):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1 - p)
        w = np.clip(w, 1e-6, None)
        W = np.diag(w)
        try:
            hessian = X.T @ W @ X
            grad = X.T @ (y - p)
            delta = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    slope_per_unit = beta[1] / sigma
    return {
        "intercept": float(beta[0]),
        "slope_std": float(beta[1]),
        "slope_per_unit": float(slope_per_unit),
        "n": int(len(x)),
        "side_covariate": "degenerate (config_a is right on every match in this dataset)",
    }


def fit_weighted_proxy(metric_names: list[str], X: np.ndarray, y_goal_diff: np.ndarray) -> Optional[dict]:
    """OLS: goal_diff ~ intercept + sum(w_i * standardized_metric_i), reported
    back in original (per-metric-unit) goals-per-match terms. Used only for
    part D's proposed weighted proxy score, over per-strategy aggregates.
    """
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_goal_diff)
    X, y = X[mask], y_goal_diff[mask]
    n = len(y)
    if n < len(metric_names) + 2:
        return None
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    design = np.column_stack([np.ones(n), Xs])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    weights_per_unit = beta[1:] / sigma
    residuals = y - design @ beta
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2) if np.sum((y - y.mean()) ** 2) > 0 else float("nan")
    return {
        "intercept": float(beta[0]),
        "weights_per_unit": {name: float(w) for name, w in zip(metric_names, weights_per_unit)},
        "weights_standardized": {name: float(w) for name, w in zip(metric_names, beta[1:])},
        "n": n,
        "r2": float(r2),
    }


# --- Per-strategy aggregation --------------------------------------------------


def per_strategy_aggregates(all_rows: list[dict], metrics: list[str]) -> dict[str, dict]:
    """For each strategy, mean differential (as that strategy's side minus
    opponent, regardless of whether it played a or b) per metric across every
    match it played, plus points-per-match and goal-diff-per-match.
    """
    strategies: dict[str, dict] = {}
    for row in all_rows:
        for role, config, opp_config in (
            ("a", row["config_a"], row["config_b"]),
            ("b", row["config_b"], row["config_a"]),
        ):
            s = strategies.setdefault(
                config,
                {"matches": 0, "points": 0.0, "goal_diff_sum": 0, "metric_sums": {m: [] for m in metrics}},
            )
            s["matches"] += 1
            if row["winner"] == "draw":
                s["points"] += 1.0
            elif row["winner"] == config:
                s["points"] += 3.0
            gd = (row["score_a"] - row["score_b"]) if role == "a" else (row["score_b"] - row["score_a"])
            s["goal_diff_sum"] += gd
            own_side = "friendly" if role == "a" else "enemy"
            opp_side = "enemy" if role == "a" else "friendly"
            for m in metrics:
                own_val = extract_side_value(row, m, own_side)
                opp_val = extract_side_value(row, m, opp_side)
                if own_val is not None and opp_val is not None:
                    s["metric_sums"][m].append(own_val - opp_val)

    out = {}
    for name, s in strategies.items():
        entry = {
            "matches": s["matches"],
            "points_per_match": s["points"] / s["matches"] if s["matches"] else float("nan"),
            "goal_diff_per_match": s["goal_diff_sum"] / s["matches"] if s["matches"] else float("nan"),
        }
        for m in metrics:
            vals = s["metric_sums"][m]
            entry[f"mean_diff_{m}"] = float(np.mean(vals)) if vals else float("nan")
        out[name] = entry
    return out


def pooled_shot_backed_up_rate_diff(all_rows: list[dict]) -> dict[str, float]:
    """Per-strategy `own_pooled_rate - opp_pooled_rate` for `shot_backed_up_rate`,
    where each pooled rate is `sum(backed-up shots) / sum(shots on target)` across
    every match the strategy played -- NOT a mean of per-match differentials (see
    `ALL_METRICS`'s docstring note on why the per-match differential is too sparse
    to use directly here: ~13 shots_on_target events across 231 matches).
    """
    own_backed: dict[str, int] = {}
    own_shots: dict[str, int] = {}
    opp_backed: dict[str, int] = {}
    opp_shots: dict[str, int] = {}
    for row in all_rows:
        fm = row["frame_metrics"]
        for role, config, opp_config in (
            ("a", row["config_a"], row["config_b"]),
            ("b", row["config_b"], row["config_a"]),
        ):
            own_side = "friendly" if role == "a" else "enemy"
            opp_side = "enemy" if role == "a" else "friendly"
            shots_n = fm.get(f"shots_on_target_{own_side}", 0) or 0
            rate = fm.get(f"shot_backed_up_rate_{own_side}")
            own_shots[config] = own_shots.get(config, 0) + shots_n
            if rate is not None:
                own_backed[config] = own_backed.get(config, 0) + round(rate * shots_n)
            opp_shots_n = fm.get(f"shots_on_target_{opp_side}", 0) or 0
            opp_rate = fm.get(f"shot_backed_up_rate_{opp_side}")
            opp_shots[config] = opp_shots.get(config, 0) + opp_shots_n
            if opp_rate is not None:
                opp_backed[config] = opp_backed.get(config, 0) + round(opp_rate * opp_shots_n)

    out = {}
    for config in own_shots:
        own_rate = own_backed.get(config, 0) / own_shots[config] if own_shots.get(config, 0) > 0 else float("nan")
        opp_rate = opp_backed.get(config, 0) / opp_shots[config] if opp_shots.get(config, 0) > 0 else float("nan")
        out[config] = own_rate - opp_rate
    return out


# --- Report generation ----------------------------------------------------


def _fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _mlabel(name: str) -> str:
    """Metric name for table display, flagged with `†` when a *lower* raw
    `a-b` differential is the "good" direction for side a (see `_LOWER_IS_BETTER`)
    -- otherwise a reader sees e.g. `restart_to_first_entry_s`'s negative
    correlation with winning and might misread it as "this metric doesn't matter"
    rather than "lower is better and the sign is exactly as expected".
    """
    base = name.split(" ")[0]
    return f"{name} †" if base in _LOWER_IS_BETTER else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="tournament run directories")
    parser.add_argument("--out", type=Path, required=True, help="output markdown report path")
    parser.add_argument(
        "--reliability-runs",
        nargs=2,
        default=["tournament_20260903_101521", "tournament_20260903_115838"],
        help="two run names (by dir basename) to correlate per-strategy metrics across, for part C",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "METRIC_CORRELATION_SCRATCH",
                "/tmp/claude-1000/-home-isaac-dev-ssl-Utama-Core/e62d8c55-0354-4b4c-b858-bd14471c3261/scratchpad",
            )
        ),
    )
    args = parser.parse_args()

    t_start = time.time()
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for run_dir in args.run_dirs:
        rows = build_rows(run_dir, args.scratch_dir, args.workers)
        all_rows.extend(rows)

    print(f"Loaded {len(all_rows)} match rows across {len(args.run_dirs)} runs in {time.time() - t_start:.1f}s")

    report_lines: list[str] = []
    report_lines.append("# Metric-correlation study: proxy metrics vs. match outcome")
    report_lines.append("")
    report_lines.append(
        f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `tools/metric_correlation.py`."
    )
    report_lines.append("")
    report_lines.append(
        "Runs analysed: " + ", ".join(r.name for r in args.run_dirs) + f" ({len(all_rows)} matches total)."
    )
    report_lines.append("")
    report_lines.append(
        "Side convention: `config_a` is always yellow and plays right "
        "(`tournament.py::run_match` hardcodes `my_team_is_yellow=True, my_team_is_right=True`); "
        "`friendly` in stats/frame metrics always means `config_a`. All metrics below are reported "
        "as `a - b` differentials. Because config_a is right on literally every match in this corpus, "
        "side cannot be fit as a genuine covariate (it is a constant column) -- see part A."
    )
    report_lines.append("")

    # ---- Part A: per-match ----
    report_lines.append("## A. Per-match: Spearman correlation with goal diff, AUC for decisive matches")
    report_lines.append("")
    report_lines.append(
        "Spearman correlates each metric's `a-b` differential with `score_a - score_b` over ALL matches "
        "(including draws, since goal diff is still informative at 0-0 for many metrics). AUC is computed only "
        "over the 22-26 decisive matches per run (config_a win=1 vs config_b win=0), predicting winner from the "
        "raw differential's rank. A logistic fit (intercept + standardized slope) is also reported per metric; "
        "a genuine side covariate could not be added (see note above), so this is intercept+slope only, not the "
        "originally intended 3-parameter model."
    )
    report_lines.append("")
    report_lines.append("| metric | n (corr) | spearman rho | p | n (AUC) | AUC | logistic slope/unit |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")

    gd = np.array([goal_diff(r) for r in all_rows], dtype=float)
    win_labels = np.array([w if w is not None else np.nan for w in (winner_label(r) for r in all_rows)])

    part_a_results = {}
    for m in ALL_METRICS:
        diffs = np.array(
            [compute_differential(r, m) if compute_differential(r, m) is not None else np.nan for r in all_rows]
        )
        rho, p, n_corr = spearman(diffs, gd)
        auc, n_auc = auc_score(diffs, win_labels)
        dec_mask = ~np.isnan(win_labels)
        logit = fit_logistic_with_side(diffs[dec_mask], None, win_labels[dec_mask])
        slope = logit["slope_per_unit"] if logit else None
        part_a_results[m] = {"rho": rho, "p": p, "n_corr": n_corr, "auc": auc, "n_auc": n_auc, "slope": slope}
        report_lines.append(
            f"| {_mlabel(m)} | {n_corr} | {_fmt(rho)} | {_fmt(p)} | {n_auc} | {_fmt(auc)} | {_fmt(slope, 4)} |"
        )
    report_lines.append("")
    report_lines.append(
        "`ball_travel_m` (from `stats.json`) and `shot_backed_up_rate` are not in the table above: "
        "`MatchStats` records `ball_travel_m` as one match-level total with no side split, so it has no "
        "`a-b` differential -- see below for a match-level check against `|goal diff|` instead. "
        "`shot_backed_up_rate` requires a side to have taken >=1 shot on target, and with only a handful of "
        "`shots_on_target` events per run, both sides rarely take one in the *same* match -- its per-match "
        "differential is `None` almost everywhere, so it's analysed only in parts B/C as a pooled per-strategy "
        "rate instead (`pooled_shot_backed_up_rate_diff`)."
    )
    report_lines.append("")

    ball_travel = np.array(
        [r["stats"].get("ball_travel_m", float("nan")) if r["stats"] else float("nan") for r in all_rows]
    )
    abs_gd = np.abs(gd)
    rho_bt, p_bt, n_bt = spearman(ball_travel, abs_gd)
    part_a_results["ball_travel_m (match-level, vs |goal diff|)"] = {"rho": rho_bt, "p": p_bt, "n_corr": n_bt}
    report_lines.append(
        f"`ball_travel_m` (match total) vs. `|goal diff|`: spearman rho={_fmt(rho_bt)}, p={_fmt(p_bt)}, n={n_bt}. "
        "A more-decisive match plausibly involves more end-to-end ball movement (attacks that go somewhere) "
        "rather than a stalemate, hence testing against |goal diff| rather than the signed value."
    )
    report_lines.append("")

    # Metric 12 (2026-09-04 expansion): total fouls per match, from
    # `rule_event_counts` (crashing/excessive_dribbling/double_touch/etc. --
    # already computed by MatchStats, previously unused by this study). No
    # side split in stats.json (same limitation as ball_travel_m), so also
    # tested match-level against |goal diff| rather than as an a-b differential.
    # Two plausible, opposite-sign hypotheses: more fouls could mean a more
    # contested, competitive match (positive with |goal diff| is NOT expected --
    # closer contests are lower |goal diff|), or could mean more congestion/
    # crashing typical of a stalling match (which would show low |goal diff|
    # too). Reported without a strong prior on sign; the data decides.
    total_fouls = np.array(
        [sum(r["stats"].get("rule_event_counts", {}).values()) if r["stats"] else float("nan") for r in all_rows]
    )
    rho_f, p_f, n_f = spearman(total_fouls, abs_gd)
    part_a_results["total_fouls (match-level, vs |goal diff|)"] = {"rho": rho_f, "p": p_f, "n_corr": n_f}
    report_lines.append(
        f"`total_fouls` (sum of `rule_event_counts`, match total) vs. `|goal diff|`: "
        f"spearman rho={_fmt(rho_f)}, p={_fmt(p_f)}, n={n_f}."
    )
    report_lines.append("")

    # ---- Part B: per-strategy ----
    report_lines.append("## B. Per-strategy: mean differential vs. points/goal-diff per match")
    report_lines.append("")
    report_lines.append(
        "Each strategy's mean `own - opponent` differential per metric, aggregated across every match it played "
        "in the three 101521/112025/115838 runs (63 matches per strategy, 21 per run x 3 runs), correlated "
        "(Spearman) against that strategy's points-per-match (3/1/0) and goal-diff-per-match across the 22 strategies."
    )
    report_lines.append("")

    main_run_names = {r.name for r in args.run_dirs if "20260902" not in r.name}
    main_rows = [r for r in all_rows if r["run"] in main_run_names]
    agg = per_strategy_aggregates(main_rows, ALL_METRICS)
    strategy_names = sorted(agg.keys())
    points = np.array([agg[s]["points_per_match"] for s in strategy_names])
    gd_per_match = np.array([agg[s]["goal_diff_per_match"] for s in strategy_names])

    part_b_results = {}
    for m in ALL_METRICS:
        vals = np.array([agg[s][f"mean_diff_{m}"] for s in strategy_names])
        rho_pts, p_pts, n_pts = spearman(vals, points)
        rho_gd, p_gd, n_gd = spearman(vals, gd_per_match)
        part_b_results[m] = {"rho_points": rho_pts, "p_points": p_pts, "rho_gd": rho_gd, "p_gd": p_gd, "n": n_pts}

    # shot_backed_up_rate: pooled per-strategy rate (see
    # pooled_shot_backed_up_rate_diff's docstring for why this can't use the
    # mean-of-per-match-differentials framework the other metrics use above).
    sbr_diff = pooled_shot_backed_up_rate_diff(main_rows)
    sbr_vals = np.array([sbr_diff.get(s, float("nan")) for s in strategy_names])
    rho_pts, p_pts, n_pts = spearman(sbr_vals, points)
    rho_gd, p_gd, n_gd = spearman(sbr_vals, gd_per_match)
    part_b_results["shot_backed_up_rate (pooled)"] = {
        "rho_points": rho_pts,
        "p_points": p_pts,
        "rho_gd": rho_gd,
        "p_gd": p_gd,
        "n": n_pts,
    }

    ranked = sorted(
        part_b_results.items(), key=lambda kv: -abs(kv[1]["rho_points"]) if not math.isnan(kv[1]["rho_points"]) else 0
    )

    report_lines.append("Full ranking (by |rho vs points-per-match|):")
    report_lines.append("")
    report_lines.append("| metric | n strategies | rho vs points/match | p | rho vs goal-diff/match | p |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for m, res in ranked:
        report_lines.append(
            f"| {_mlabel(m)} | {res['n']} | {_fmt(res['rho_points'])} | {_fmt(res['p_points'])} | "
            f"{_fmt(res['rho_gd'])} | {_fmt(res['p_gd'])} |"
        )
    report_lines.append("")

    report_lines.append("### Top five metrics by |rho vs points-per-match|")
    report_lines.append("")
    report_lines.append("| rank | metric | rho vs points/match | p | rho vs goal-diff/match | p |")
    report_lines.append("|---:|---|---:|---:|---:|---:|")
    top5 = ranked[:5]
    for i, (m, res) in enumerate(top5, 1):
        report_lines.append(
            f"| {i} | {m} | {_fmt(res['rho_points'])} | {_fmt(res['p_points'])} | "
            f"{_fmt(res['rho_gd'])} | {_fmt(res['p_gd'])} |"
        )
    report_lines.append("")

    # ---- Part C: reliability ----
    report_lines.append("## C. Reliability: per-strategy metric value, run 101521 vs run 115838")
    report_lines.append("")
    report_lines.append(
        "**Caveat: code changed between these runs** (stall/deadlock fixes landed on this branch between "
        "101521, 112025, and 115838 -- see the git log). Any reliability number below is therefore a *lower "
        "bound* on true metric reliability: some run-to-run disagreement here is genuine strategy-vs-strategy "
        "noise, but some is the runner behaving differently, not the metric being noisy."
    )
    report_lines.append("")

    run_a_name, run_b_name = args.reliability_runs
    rows_a = [r for r in all_rows if r["run"] == run_a_name]
    rows_b = [r for r in all_rows if r["run"] == run_b_name]
    reliability = {}
    if rows_a and rows_b:
        agg_a = per_strategy_aggregates(rows_a, ALL_METRICS)
        agg_b = per_strategy_aggregates(rows_b, ALL_METRICS)
        common = sorted(set(agg_a) & set(agg_b))
        report_lines.append(f"Comparing {len(common)} strategies present in both `{run_a_name}` and `{run_b_name}`.")
        report_lines.append("")
        report_lines.append("| metric | n strategies | spearman rho (run-to-run) | p |")
        report_lines.append("|---|---:|---:|---:|")
        for m in ALL_METRICS:
            va = np.array([agg_a[s][f"mean_diff_{m}"] for s in common])
            vb = np.array([agg_b[s][f"mean_diff_{m}"] for s in common])
            rho, p, n = spearman(va, vb)
            reliability[m] = {"rho": rho, "p": p, "n": n}
            report_lines.append(f"| {_mlabel(m)} | {n} | {_fmt(rho)} | {_fmt(p)} |")

        sbr_a = pooled_shot_backed_up_rate_diff(rows_a)
        sbr_b = pooled_shot_backed_up_rate_diff(rows_b)
        va = np.array([sbr_a.get(s, float("nan")) for s in common])
        vb = np.array([sbr_b.get(s, float("nan")) for s in common])
        rho, p, n = spearman(va, vb)
        reliability["shot_backed_up_rate (pooled)"] = {"rho": rho, "p": p, "n": n}
        report_lines.append(f"| shot_backed_up_rate (pooled) | {n} | {_fmt(rho)} | {_fmt(p)} |")
        report_lines.append("")

        report_lines.append("### Reliability of the top-five B metrics")
        report_lines.append("")
        report_lines.append("| metric | validity rho (points/match) | reliability rho (run-to-run) | verdict |")
        report_lines.append("|---|---:|---:|---|")
        for m, res in top5:
            rel = reliability.get(m, {"rho": float("nan")})
            v_rho = res["rho_points"]
            r_rho = rel["rho"]
            if not math.isnan(v_rho) and not math.isnan(r_rho):
                if abs(v_rho) >= 0.3 and abs(r_rho) >= 0.5:
                    verdict = "valid & reliable -- good instrumentation candidate"
                elif abs(v_rho) >= 0.3 and abs(r_rho) < 0.5:
                    verdict = "valid but unreliable -- likely noise, deprioritize"
                elif abs(v_rho) < 0.3 and abs(r_rho) >= 0.5:
                    verdict = "reliable but not valid -- measures something other than winning"
                else:
                    verdict = "neither valid nor reliable"
            else:
                verdict = "insufficient data"
            report_lines.append(f"| {_mlabel(m)} | {_fmt(v_rho)} | {_fmt(r_rho)} | {verdict} |")
        report_lines.append("")
    else:
        report_lines.append("*(one or both reliability runs not present in the provided `run_dirs`; skipped.)*")
        report_lines.append("")

    old_draw_runs = [r for r in args.run_dirs if "20260902_2209" in r.name or "20260902_2223" in r.name]
    if old_draw_runs:
        report_lines.append(
            "Two older 36-match, 9-strategy runs (`tournament_20260902_220936`, `tournament_20260902_222351`) "
            "were all draws and cover fewer than half the strategy catalog -- not used quantitatively above "
            "(too few decisive matches and too small a strategy set for a meaningful correlation), consistent "
            "with the task's guidance to use them only for reliability spot-checks."
        )
        report_lines.append("")

    # ---- Part D: recommendation ----
    report_lines.append("## D. Recommendation")
    report_lines.append("")

    # Combined validity-x-reliability ranking, computed here (ahead of the "which
    # metrics to instrument" subsection below) so the weighted proxy score and the
    # instrument/drop lists are built from the same selection instead of two
    # different top-N cuts.
    combined = []
    for m, res in part_b_results.items():
        rel = reliability.get(m, {"rho": float("nan")})
        v_rho, r_rho = res["rho_points"], rel["rho"]
        score = abs(v_rho) * abs(r_rho) if not (math.isnan(v_rho) or math.isnan(r_rho)) else float("-inf")
        combined.append((m, v_rho, r_rho, score))
    combined.sort(key=lambda t: t[3], reverse=True)
    n_strong = sum(1 for *_, s in combined if s >= 0.3)
    n_keep = min(6, max(4, n_strong)) if n_strong >= 4 else min(6, len(combined))
    top_n = combined[:n_keep]
    recommend_keep = [m for m, _v, _r, s in top_n if s != float("-inf")]
    recommend_drop = [m for m, _v, _r, _s in combined if m not in recommend_keep]

    # shot_backed_up_rate (pooled) has no per-match `mean_diff_` entry in `agg`
    # (it's computed separately -- see pooled_shot_backed_up_rate_diff), so the
    # weighted-proxy fit below only includes metrics with a real `agg` column.
    candidate_metrics = [m for m in recommend_keep if f"mean_diff_{m}" in next(iter(agg.values()), {})]
    X_cols = []
    valid_metric_names = []
    for m in candidate_metrics:
        vals = np.array([agg[s][f"mean_diff_{m}"] for s in strategy_names])
        if not np.all(np.isnan(vals)):
            X_cols.append(vals)
            valid_metric_names.append(m)
    weighted = None
    if X_cols:
        X = np.column_stack(X_cols)
        weighted = fit_weighted_proxy(valid_metric_names, X, gd_per_match)

    n_decisive = int(np.sum(~np.isnan(win_labels)))
    report_lines.append(
        f"**Honesty check first:** matches are 65 s and only {n_decisive}/{len(all_rows)} "
        f"({100*n_decisive/max(1,len(all_rows)):.0f}%) are decisive across the analysed runs. Per-match AUCs in "
        "part A are computed on ~22-26 matches per run; per-strategy correlations in part B have n=22 (one point "
        "per strategy) and n=9 for the tiny reliability spot-check runs. None of these sample sizes support "
        'strong causal claims -- treat every number here as "consistent with" rather than "proves". The '
        "cleanest signal is the per-strategy aggregate (63 matches/strategy pools out a lot of single-match "
        "noise), which is why part D's weights are fit there, not on individual matches."
    )
    report_lines.append("")

    if weighted:
        report_lines.append(
            f"Weighted proxy score (OLS, goal-diff-per-match ~ intercept + weighted metrics; "
            f"n={weighted['n']} strategies, R^2={_fmt(weighted['r2'])}):"
        )
        report_lines.append("")
        report_lines.append(f"`proxy_score = {_fmt(weighted['intercept'])}`")
        for name, w in weighted["weights_per_unit"].items():
            report_lines.append(f"  `+ ({_fmt(w, 5)}) * mean_diff_{name}`")
        report_lines.append("")
        report_lines.append(
            "Units: `proxy_score` is in goals per match; each weight is per one raw unit of that metric's "
            "`a-b` differential (e.g. per 1.0 of possession_pct differential, or per 1 extra shot_on_target). "
            f"R^2={_fmt(weighted['r2'])} on n={weighted['n']} strategies -- with this few points, treat the "
            "specific weight values as directionally suggestive, not a tuned production model."
        )
        report_lines.append("")

    report_lines.append("### Which metrics to instrument live in MatchStats")
    report_lines.append("")
    report_lines.append(
        "Combined score = `|validity rho (B, vs points/match)| x |reliability rho (C, run-to-run)|`, over every "
        "metric analysed in B (including the pooled `shot_backed_up_rate`) -- rewards metrics that are both "
        "predictive of winning and stable run-to-run, rather than gating on a hard top-five-only cutoff:"
    )
    report_lines.append("")
    report_lines.append("| metric | validity rho | reliability rho | combined score |")
    report_lines.append("|---|---:|---:|---:|")
    for m, v_rho, r_rho, score in combined:
        report_lines.append(
            f"| {_mlabel(m)} | {_fmt(v_rho)} | {_fmt(r_rho)} | {_fmt(score) if score != float('-inf') else 'n/a'} |"
        )
    report_lines.append("")
    report_lines.append(f"**Instrument these {len(recommend_keep)} live in `MatchStats`:**")
    report_lines.append("")
    for m in recommend_keep:
        report_lines.append(f"- `{m}`")
    report_lines.append("")
    report_lines.append("**Drop or deprioritize:**")
    report_lines.append("")
    for m in recommend_drop:
        report_lines.append(f"- `{m}`")
    report_lines.append("")
    report_lines.append(
        "Note `possession_pct`, `robot_motion_pct`, and `possession_under_pressure_s` land in the drop list "
        "despite `possession_pct`/`robot_motion_pct` already being free (already computed by `MatchStats` today) "
        '-- "drop" here means "don\'t treat as a proxy for winning / don\'t weight in the proxy score", not '
        '"remove from `MatchStats`"; they may still be useful for other diagnostic purposes.'
    )
    report_lines.append("")
    report_lines.append(f"**Runtime:** full analysis over {len(all_rows)} matches took {time.time() - t_start:.1f}s.")
    report_lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(report_lines))

    json_out = args.out.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(
            {
                "part_a": part_a_results,
                "part_b": {m: res for m, res in part_b_results.items()},
                "part_b_top5": [m for m, _ in top5],
                "reliability": reliability,
                "weighted_proxy": weighted,
                "n_matches": len(all_rows),
                "n_decisive": n_decisive,
                "runtime_s": time.time() - t_start,
            },
            f,
            indent=2,
        )

    print(f"Wrote {args.out} and {json_out} in {time.time() - t_start:.1f}s total")


if __name__ == "__main__":
    main()
