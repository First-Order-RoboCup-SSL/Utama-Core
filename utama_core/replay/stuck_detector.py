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

from utama_core.entities.game import GameFrame
from utama_core.replay.replay_player import _load_replay


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


def find_stuck_windows(
    replay_path: Union[str, Path],
    *,
    window_s: float = 3.0,
    stride_s: float = 1.0,
    ball_still_tol: float = 0.05,
    oscillation_energy_tol: float = 0.8,
    min_duration_s: float = 3.0,
) -> list[StuckWindow]:
    """Slide a `window_s`-wide window (every `stride_s`) across a replay and
    flag windows where the ball is frozen (`ball_std < ball_still_tol`) and
    at least one friendly robot's position trace has non-DC spectral energy
    fraction above `oscillation_energy_tol`.

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

    raw_windows: list[StuckWindow] = []
    t = t0
    while t + window_s <= t_last:
        window_frames = [f for f in frames if t <= f.ts <= t + window_s]
        if len(window_frames) >= 4 and all(f.ball is not None for f in window_frames):
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
