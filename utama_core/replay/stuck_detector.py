"""Offline detector for "stuck" windows in a replay: the ball frozen while a
robot oscillates instead of making progress.

This is a prototype analysis tool, not a live `custom_referee` rule (see
`docs/testing_gaps.md` gap #11) — it runs after a match, over a recorded
`.pkl` replay, and reports candidate windows for a human/agent to look at
with `render_window()`. It intentionally does not (and should not, yet) feed
back into any in-match decision: a false "stuck" call inside a live rule
would be exactly the class of referee bug tracked as gap #9 in the same
doc, so this stays an offline diagnostic until validated against real
replays.

Two independent per-window signals, both required to flag a window:

- **Ball frozen**: the ball's position standard deviation over the window
  is below `ball_still_tol` (metres). A ball that's genuinely in play
  (being dribbled, passed, rolling to a stop) moves more than this within
  any multi-second window; one that's stuck (wedged against a robot,
  sitting in a corner while nothing resolves it) doesn't.
- **A robot oscillating, not converging**: for each robot's x and y
  position trace over the window, take the FFT, and look at the energy in
  frequencies above `min_oscillation_hz` relative to total energy (excluding
  DC). A robot settling into a hold position has its energy concentrated
  near DC; a robot flip-flopping (e.g. two robots endlessly contesting one
  point, or a path planner alternating detour sides — see the goalkeeper
  bug this same investigation found, `docs/roadmap.md`'s "Goalkeeper
  overshoot" entry) shows up as a real spectral peak away from DC.

Both conditions together, sustained for `min_duration_s`, are what
distinguish a genuine stuck match from an ordinary contested-ball moment
(ball frozen alone also happens at every legal stoppage; oscillation alone
also happens during normal marking/jostling) — see gap #11 in
`docs/testing_gaps.md` for the reasoning and the "run this over the gap #6
replays first" validation plan before ever wiring this into live play.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.entities.game import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.replay_player import _load_replay

# Referee commands where the ball is expected to be legally stationary (a
# stoppage, restart ceremony, or placement) — a window spent mostly in one
# of these is not a "stuck match" in gap #11's sense, just normal officiating.
_LIVE_PLAY_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})


@dataclass(frozen=True)
class StuckWindow:
    """One candidate stuck window found in a replay."""

    t_start: float
    t_end: float
    ball_std: float
    """Ball position std-dev (metres) over the window — low means frozen."""
    oscillating_robot_ids: tuple[int, ...]
    """Friendly-team robot ids whose x/y trace showed non-DC spectral energy."""


def _dominant_non_dc_fraction(trace: np.ndarray) -> float:
    """Fraction of a 1D signal's spectral energy concentrated in its single
    strongest non-DC frequency bin.

    Near 0 for a settled hold or a monotonic transient (a decaying
    approach, a one-way drift) — both spread their (small) non-DC energy
    thinly across many bins, since neither is periodic. Near 1 for a
    genuinely oscillating signal (energy concentrated at one repeating
    frequency) — e.g. two robots flip-flopping around a contested point,
    or a path planner alternating detour sides. `trace` is assumed evenly
    sampled (true for a fixed-tick-rate replay). Looking for a *peak*
    rather than "any non-DC energy" is what tells a real oscillation apart
    from ordinary spectral leakage off a sharp but non-repeating motion —
    a flat "sum of non-DC energy" measure flags both alike.
    """
    n = len(trace)
    if n < 4:
        return 0.0
    centered = trace - trace.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    non_dc = spectrum[1:]
    total = non_dc.sum()
    if total <= 1e-12:
        return 0.0
    return float(non_dc.max() / total)


def _in_defense_area(x: float, y: float, field_dims: FieldDimensions) -> bool:
    depth = field_dims.half_defense_area_depth * 2
    half_width = field_dims.half_defense_area_width
    half_length = field_dims.full_field_half_length
    in_left = x <= -half_length + depth and abs(y) <= half_width
    in_right = x >= half_length - depth and abs(y) <= half_width
    return in_left or in_right


def find_stuck_windows(
    replay_path: Union[str, Path],
    *,
    window_s: float = 3.0,
    stride_s: float = 1.0,
    ball_still_tol: float = 0.05,
    oscillation_energy_tol: float = 0.8,
    min_duration_s: float = 3.0,
    live_play_fraction: float = 0.9,
    possession_fraction: float = 0.3,
    defense_area_fraction: float = 0.5,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
) -> list[StuckWindow]:
    """Slide a `window_s`-wide window (every `stride_s`) across a replay and
    flag windows where the ball is frozen (`ball_std < ball_still_tol`) and
    at least one friendly robot's position trace has non-DC spectral energy
    fraction above `oscillation_energy_tol`.

    Three false-positive classes are excluded before the frozen/oscillating
    check even runs, all found via a 2026-09-02 sweep of a competitive
    tournament re-run (every one of that run's "genuine" raw-flagged windows
    turned out to be one of these three, not a real stuck-match bug):

    - **Not live play.** A window where the referee command is
      `NORMAL_START`/`FORCE_START` for less than `live_play_fraction` of its
      frames is a legal stoppage/restart (kickoff standstill, a mid-match
      restart after a goal, a `STOP`<->`FORCE_START` violation cycle), not a
      stuck match — the ball is *supposed* to be frozen there. Replays with
      no referee data (`frame.referee is None`) skip this check entirely
      rather than being unflaggable by construction, since not every replay
      (e.g. a unit-test fixture, or a non-refereed `debug_match.py` run)
      carries referee state.
    - **Ball possessed.** A window where some robot (either team) has
      `has_ball=True` for at least `possession_fraction` of its frames is a
      robot legitimately holding/shielding the ball, not a stall — this is
      the majority case in practice (e.g. a carrier paused mid-decision).
    - **Ball in a defense area.** A window where the ball spends at least
      `defense_area_fraction` of its frames inside either team's defense
      box is already governed by the referee's own held-ball/interference
      rules (which resolve it via `STOP`/`BALL_PLACEMENT` on their own
      timeout, distinct from and faster than this detector's), so flagging
      it here would just be re-reporting a case the referee already handles.

    Adjacent/overlapping flagged windows are merged before returning, and
    merged spans shorter than `min_duration_s` are dropped — a single
    flagged 3s window on its own is exactly `window_s`, so `min_duration_s`
    only starts filtering once windows are tuned to overlap more (smaller
    `stride_s`) or `window_s` itself is shortened.
    """
    frames: list[GameFrame] = [obj for obj in _load_replay(replay_path) if isinstance(obj, GameFrame)]
    if not frames:
        return []

    t0 = frames[0].ts
    t_last = frames[-1].ts

    # `frames` is time-ordered, so each window's frame slice can be found by
    # advancing two indices rather than rescanning the whole list from t0 on
    # every stride step. Both indices only ever move forward across the
    # entire sweep (never reset between iterations), since consecutive
    # windows' start/end times are non-decreasing — this makes the total
    # index movement O(n_frames) instead of the previous O(n_frames *
    # n_strides), the dominant cost for a full-length (600s @ 60Hz = 36k
    # frames) replay. Output is unchanged: `frames[start:end]` is exactly
    # the same frame set the old `[f for f in frames if t <= f.ts <= t +
    # window_s]` filter produced, since both select on the same half-open
    # condition over an already-sorted sequence.
    start_idx = 0
    end_idx = 0
    n = len(frames)

    raw_windows: list[StuckWindow] = []
    t = t0
    while t + window_s <= t_last:
        while start_idx < n and frames[start_idx].ts < t:
            start_idx += 1
        if end_idx < start_idx:
            end_idx = start_idx
        while end_idx < n and frames[end_idx].ts <= t + window_s:
            end_idx += 1
        window_frames = frames[start_idx:end_idx]
        if len(window_frames) >= 4 and all(f.ball is not None for f in window_frames):
            if window_frames[0].referee is not None:
                live_frac = sum(
                    f.referee.referee_command in _LIVE_PLAY_COMMANDS for f in window_frames if f.referee
                ) / len(window_frames)
                if live_frac < live_play_fraction:
                    t += stride_s
                    continue

            held_frac = sum(
                any(r.has_ball for r in f.friendly_robots.values()) or any(r.has_ball for r in f.enemy_robots.values())
                for f in window_frames
            ) / len(window_frames)
            if held_frac >= possession_fraction:
                t += stride_s
                continue

            defense_frac = sum(_in_defense_area(f.ball.p.x, f.ball.p.y, field_dims) for f in window_frames) / len(
                window_frames
            )
            if defense_frac >= defense_area_fraction:
                t += stride_s
                continue

            ball_xs = np.array([f.ball.p.x for f in window_frames])
            ball_ys = np.array([f.ball.p.y for f in window_frames])
            ball_std = float(np.hypot(ball_xs.std(), ball_ys.std()))

            if ball_std < ball_still_tol:
                oscillating: list[int] = []
                robot_ids = set.intersection(*(set(f.friendly_robots) for f in window_frames))
                for rid in sorted(robot_ids):
                    xs = np.array([f.friendly_robots[rid].p.x for f in window_frames])
                    ys = np.array([f.friendly_robots[rid].p.y for f in window_frames])
                    energy = max(_dominant_non_dc_fraction(xs), _dominant_non_dc_fraction(ys))
                    if energy > oscillation_energy_tol:
                        oscillating.append(rid)

                if oscillating:
                    raw_windows.append(
                        StuckWindow(
                            t_start=t,
                            t_end=t + window_s,
                            ball_std=ball_std,
                            oscillating_robot_ids=tuple(oscillating),
                        )
                    )
        t += stride_s

    return _merge_windows(raw_windows, min_duration_s=min_duration_s)


def _merge_windows(windows: list[StuckWindow], *, min_duration_s: float) -> list[StuckWindow]:
    if not windows:
        return []

    merged: list[StuckWindow] = []
    current = windows[0]
    for w in windows[1:]:
        if w.t_start <= current.t_end:
            current = StuckWindow(
                t_start=current.t_start,
                t_end=max(current.t_end, w.t_end),
                ball_std=max(current.ball_std, w.ball_std),
                oscillating_robot_ids=tuple(sorted(set(current.oscillating_robot_ids) | set(w.oscillating_robot_ids))),
            )
        else:
            merged.append(current)
            current = w
    merged.append(current)

    return [w for w in merged if (w.t_end - w.t_start) >= min_duration_s]
