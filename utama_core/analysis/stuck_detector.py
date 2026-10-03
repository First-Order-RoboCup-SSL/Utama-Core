"""Offline detector for "stuck" windows in a replay: the ball frozen while a
robot oscillates instead of making progress, or a restart command stuck in
effect far longer than any legal restart ceremony takes.

This is a prototype analysis tool, not a live `custom_referee` rule (see
`docs/testing_gaps.md` gap #11) — it runs after a match, over a recorded
`.pkl`/`.npz` replay, and reports candidate windows for a human/agent to
look at with `render_window()`. It intentionally does not (and should not,
yet) feed back into any in-match decision: a false "stuck" call inside a
live rule would be exactly the class of referee bug tracked as gap #9 in
the same doc, so this stays an offline diagnostic until validated against
real replays.

Two independent detection classes, each producing a `StuckWindow` with a
distinct `kind`:

- **`kind="oscillation"`** — two independent per-window signals, both
  required to flag a window:
  - **Ball frozen**: the ball's position standard deviation over the
    window is below `ball_still_tol` (metres). A ball that's genuinely in
    play (being dribbled, passed, rolling to a stop) moves more than this
    within any multi-second window; one that's stuck (wedged against a
    robot, sitting in a corner while nothing resolves it) doesn't.
  - **A robot oscillating, not converging**: for each robot's x and y
    position trace over the window, take the FFT, and look at the energy
    in frequencies above `min_oscillation_hz` relative to total energy
    (excluding DC). A robot settling into a hold position has its energy
    concentrated near DC; a robot flip-flopping (e.g. two robots endlessly
    contesting one point, or a path planner alternating detour sides — see
    the goalkeeper bug this same investigation found, `docs/roadmap.md`'s
    "Goalkeeper overshoot" entry) shows up as a real spectral peak away
    from DC.

  Both conditions together, sustained for `min_duration_s`, are what
  distinguish a genuine stuck match from an ordinary contested-ball moment
  (ball frozen alone also happens at every legal stoppage; oscillation
  alone also happens during normal marking/jostling) — see gap #11 in
  `docs/testing_gaps.md` for the reasoning and the "run this over the gap
  #6 replays first" validation plan before ever wiring this into live
  play.

- **`kind="restart_stall"`** — a restart command (anything but
  `NORMAL_START`/`FORCE_START`/`HALT`) held for more than `restart_stall_s`
  (default 15s) with the ball never moving: no legal restart ceremony
  takes that long, so this is a stall the referee never auto-advanced out
  of, not a legal stoppage. Added after a real match
  (`high_line_zone_vs_overload_flow`, 2026-09-03) stalled in
  `DIRECT_FREE_BLUE` for the rest of the match — a free-kick taker
  crawling toward the ball too slowly to ever reach kicker-ready distance
  — and the oscillation detector above returned zero windows for it: the
  "not live play" exclusion (correctly) couldn't tell a legal restart from
  a stalled one on its own, and the "ball possessed" exclusion doesn't
  apply here at all (nobody has the ball yet). See `find_stuck_windows`'s
  docstring for the full mechanism, including the two other cases it
  fixed at the same time (a replay with no `frame.referee` data falling
  back to the `<name>.intentions.jsonl` sidecar; a possession exclusion
  that no longer applies when the possessor itself is motionless).
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.entities.game import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_reader import ColumnarReplay, load_columnar_replay
from utama_core.replay.replay_player import _load_replay

# Referee commands where the ball is expected to be legally stationary (a
# stoppage, restart ceremony, or placement) — a window spent mostly in one
# of these is not a "stuck match" in gap #11's sense, just normal officiating.
_LIVE_PLAY_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})

# Commands where the ball being frozen is *expected but time-limited*: a
# restart ceremony (free kick, kickoff, penalty, ball placement, timeout)
# gives the taker a bounded amount of real time to act before it stops being
# "normal officiating" and starts being a stall nothing is auto-advancing —
# see `RESTART_STALL` below. HALT is excluded: the referee/operator has
# deliberately frozen the match, which is not a "stall" this detector should
# ever flag.
_RESTART_COMMANDS = frozenset(RefereeCommand) - _LIVE_PLAY_COMMANDS - {RefereeCommand.HALT}


class RefereeSidecar:
    """Reconstructs a per-timestamp referee command from a replay's
    `<name>.intentions.jsonl` sidecar, for replays whose `GameFrame`/
    columnar data carries no referee state at all (every tournament replay
    written so far: `frame.referee is None`/`has_referee[...] == False` for
    every tick — the tournament replay writer doesn't carry referee state
    into the `.pkl`/`.npz` itself). The sidecar's `"event": "referee"` lines
    each carry `sim_time` (when the command took effect) and `command` (the
    `RefereeCommand` enum name as a string) — the command is a step
    function of time: whatever the most recent `"referee"` event at or
    before a given `sim_time` set it to, held until the next event.

    Best-effort: a replay with no matching sidecar file, or one that can't
    be parsed, yields `command_at` returning `None` for every timestamp,
    which callers treat exactly like "no referee data available" (the
    existing behaviour for a replay with genuinely no referee state).
    """

    def __init__(self, sim_times: np.ndarray, commands: list[Optional[RefereeCommand]]):
        self._sim_times = sim_times
        self._commands = commands

    @property
    def available(self) -> bool:
        return len(self._sim_times) > 0

    def command_at(self, ts: float) -> Optional[RefereeCommand]:
        if not self._sim_times.size:
            return None
        idx = bisect_right(self._sim_times, ts) - 1
        if idx < 0:
            return None
        return self._commands[idx]

    @staticmethod
    def for_replay(replay_path: Path) -> "RefereeSidecar":
        sidecar_path = replay_path.with_suffix("").with_suffix(".intentions.jsonl")
        sim_times: list[float] = []
        commands: list[Optional[RefereeCommand]] = []
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if entry.get("event") != "referee":
                            continue
                        command_name = entry.get("command")
                        if command_name is None:
                            continue
                        try:
                            command = RefereeCommand[command_name]
                        except KeyError:
                            continue
                        sim_times.append(float(entry["sim_time"]))
                        commands.append(command)
            except (OSError, json.JSONDecodeError):
                return RefereeSidecar(np.array([]), [])
        # Events are appended in order as the match runs, so `sim_times` is
        # already non-decreasing — no sort needed for `bisect_right` to work.
        return RefereeSidecar(np.array(sim_times), commands)


@dataclass(frozen=True)
class StuckWindow:
    """One candidate stuck window found in a replay."""

    t_start: float
    t_end: float
    ball_std: float
    """Ball position std-dev (metres) over the window — low means frozen."""
    oscillating_robot_ids: tuple[int, ...]
    """Friendly-team robot ids whose x/y trace showed non-DC spectral energy.
    Always empty for a `kind="restart_stall"` window — that class doesn't
    depend on oscillation at all, see `find_stuck_windows`."""
    kind: str = "oscillation"
    """Which stall signal flagged this window: `"oscillation"` (the
    original ball-frozen + robot-oscillating signal) or `"restart_stall"`
    (a non-live-play referee command held far longer than any restart
    ceremony should take, with the ball frozen throughout — see
    `find_stuck_windows`). Defaults to `"oscillation"` so existing callers
    unpacking/constructing a `StuckWindow` positionally or via the first
    four fields keep working unchanged."""


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


_CONTINUOUS_POSSESSION_FRACTION = 0.95
"""How much of a window a single robot must hold the ball for before
`_possessor_is_motionless` will even consider bypassing the possession
exclusion. Deliberately much stricter than `possession_fraction` (0.3):
a robot legitimately shielding/holding the ball while teammates or
opponents jostle for position around it — the common case the possession
exclusion exists for, `possession_fraction`'s own docstring's "the
majority case in practice" — routinely holds the ball for a large minority
of a window while sitting nearly still (see the `clear_danger_vs_high_press`
finding this constant was added to fix: an enemy robot held the ball
motionless for 87% of a window while two unrelated friendly robots
genuinely converged/marked around it — a real find, not a stall). What
distinguishes the actual target case, a carrier parked on a frozen ball for
the *entire* window with nothing else resolving it, is *continuous*, not
just frequent, possession by the same robot — hence the near-100% bar
here rather than reusing `possession_fraction`.
"""

_DEFAULT_POSSESSOR_STILL_MIN_S = 15.0
"""Default `possessor_still_min_s`: how long a single robot must hold the
ball *continuously* (spanning, not just within, one `window_s`-wide
window) before `_possessor_is_motionless` bypasses the possession
exclusion at all. Added after `_CONTINUOUS_POSSESSION_FRACTION` alone
(a 2026-09-03 calibration sweep over every replay under `replays/`,
`ball_travel_m` from each match's `summary.json`/`.stats.json` as ground
truth) turned out to fire on any ordinary short ball-control moment — a
kickoff-taker settling the ball before a pass, a receiver controlling a
pass — that happens to hold still for a couple of seconds: e.g.
`clear_danger_vs_clear_press_plus.pkl` had robot `friendly/1` sitting on
the ball motionless from t=5.17s to t=27.0s after a completely ordinary
kickoff, which the window-local check alone flagged from its first
`window_s`-wide slice at t=10s — 5s into a hold that turned out to
resolve on its own 17s later, nowhere near an unresolved stall. Sampling
continuous single-robot possession runs across 20 real matches showed
ordinary holds top out around 14s and genuine stuck-on-the-ball runs start
around 41s — a wide gap — so the floor is set to 15s, matching
`restart_stall_s`'s own "no legal ceremony/touch takes this long" bar,
rather than introducing a second unrelated constant.
"""


def _possession_runs(
    possessor_at: list[Optional[tuple[str, int]]], ts: np.ndarray
) -> list[tuple[int, int, tuple[str, int]]]:
    """Collapse a per-tick "which single robot (if any) exclusively holds
    the ball" sequence into maximal `(start_idx, end_idx, key)` runs of the
    same non-`None` possessor at consecutive ticks — the same "maximal run
    of a repeated per-tick condition" shape `_find_restart_stalls` uses for
    referee commands, applied here to possession identity so
    `_possessor_is_motionless`/`_possessor_is_motionless_columnar` can
    check a window against the *real* duration of the hold it sits inside,
    not just what's visible within that one window (see
    `_DEFAULT_POSSESSOR_STILL_MIN_S`). `end_idx` is exclusive, matching
    Python slice convention (`possessor_at[start_idx:end_idx]` is the run).
    """
    runs: list[tuple[int, int, tuple[str, int]]] = []
    start = None
    current: Optional[tuple[str, int]] = None
    for i, key in enumerate(possessor_at):
        if key != current:
            if current is not None:
                runs.append((start, i, current))
            current = key
            start = i if key is not None else None
    if current is not None:
        runs.append((start, len(possessor_at), current))
    return runs


def _possessor_is_motionless(
    window_frames: list[GameFrame],
    possessor_still_tol: float,
    *,
    window_start_idx: int,
    window_end_idx: int,
    possession_runs: list[tuple[int, int, tuple[str, int]]],
    possessor_still_min_s: float,
    ts: np.ndarray,
) -> bool:
    """True when a single robot holds the ball (`has_ball=True`) for at
    least `_CONTINUOUS_POSSESSION_FRACTION` of `window_frames`, has
    position std-dev below `possessor_still_tol` over those frames, AND
    that same robot's continuous hold (from `possession_runs`, computed
    once over the whole replay — see `_possession_runs`) spans at least
    `possessor_still_min_s` real seconds, not just this one window. A
    carrier parked on a frozen ball for essentially the whole window is
    the stall this detector exists to catch (see `find_stuck_windows`'s
    possession exclusion docs); a carrier genuinely dribbling/
    repositioning, one that only holds the ball for part of a window while
    normal contested play continues around it (the majority real-world
    case the possession exclusion is for), or one that merely holds still
    for a normal few-second ball-control moment (the false positive
    `_DEFAULT_POSSESSOR_STILL_MIN_S` was added to fix — see its
    docstring), is not, and keeps the exclusion.
    """
    possessor_xs: dict[tuple[str, int], list[float]] = {}
    possessor_ys: dict[tuple[str, int], list[float]] = {}
    for f in window_frames:
        for team, robots in (("friendly", f.friendly_robots), ("enemy", f.enemy_robots)):
            for rid, r in robots.items():
                if r.has_ball:
                    key = (team, rid)
                    possessor_xs.setdefault(key, []).append(r.p.x)
                    possessor_ys.setdefault(key, []).append(r.p.y)
    n = len(window_frames)
    for key, xs in possessor_xs.items():
        if len(xs) / n < _CONTINUOUS_POSSESSION_FRACTION:
            continue
        ys = possessor_ys[key]
        std = float(np.hypot(np.std(xs), np.std(ys)))
        if std >= possessor_still_tol:
            continue
        if _run_duration_covering(possession_runs, key, window_start_idx, window_end_idx, ts) >= possessor_still_min_s:
            return True
    return False


def _run_duration_covering(
    possession_runs: list[tuple[int, int, tuple[str, int]]],
    key: tuple[str, int],
    window_start_idx: int,
    window_end_idx: int,
    ts: np.ndarray,
) -> float:
    """Duration (seconds) of the possession run in `possession_runs` for
    `key` that overlaps `[window_start_idx, window_end_idx)`, or 0.0 if
    none does. A window can only overlap one run of a given possessor
    (runs are maximal and non-overlapping by construction), so the first
    match found is the answer.
    """
    for start, end, run_key in possession_runs:
        if run_key != key:
            continue
        if start < window_end_idx and end > window_start_idx:
            return float(ts[end - 1] - ts[start])
    return 0.0


def _find_restart_stalls(
    *,
    ts: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    commands: list[Optional[RefereeCommand]],
    ball_still_tol: float,
    restart_stall_s: float,
) -> list[StuckWindow]:
    """Single linear pass over a replay's full tick sequence: find maximal
    runs where `commands[i]` is a restart command (`_RESTART_COMMANDS` —
    anything but live play or `HALT`) and the ball stays within
    `ball_still_tol` of the run's starting position, and flag any run
    lasting more than `restart_stall_s`. See `find_stuck_windows`'s
    "Restart stall" docs for why this is a separate pass rather than a
    variant of the sliding-window sweep: it has no oscillation
    requirement, and a legal restart's normal duration (a few seconds) is
    far shorter than `window_s`-scale windowing would resolve well.

    A run breaks (and a new one starts) whenever the command changes,
    ball position departs more than `ball_still_tol` from the run's start,
    or the command is unknown (`None` — no referee data and no usable
    sidecar entry for that tick), matching the rest of this module's
    "skip the check rather than guess" behaviour for replays with no
    referee information at all.
    """
    n = len(ts)
    windows: list[StuckWindow] = []
    if n == 0:
        return windows

    run_start_idx: Optional[int] = None
    run_start_x = run_start_y = 0.0

    def _close_run(end_idx: int) -> None:
        nonlocal run_start_idx
        if run_start_idx is None:
            return
        t_start = float(ts[run_start_idx])
        t_end = float(ts[end_idx - 1])
        if t_end - t_start >= restart_stall_s:
            xs = ball_x[run_start_idx:end_idx]
            ys = ball_y[run_start_idx:end_idx]
            ball_std = float(np.hypot(np.nanstd(xs), np.nanstd(ys)))
            windows.append(
                StuckWindow(
                    t_start=t_start,
                    t_end=t_end,
                    ball_std=ball_std,
                    oscillating_robot_ids=(),
                    kind="restart_stall",
                )
            )
        run_start_idx = None

    for i in range(n):
        cmd = commands[i]
        bx, by = ball_x[i], ball_y[i]
        in_restart = cmd in _RESTART_COMMANDS and not np.isnan(bx) and not np.isnan(by)

        if not in_restart:
            _close_run(i)
            continue

        if run_start_idx is None:
            run_start_idx = i
            run_start_x, run_start_y = bx, by
            continue

        if np.hypot(bx - run_start_x, by - run_start_y) > ball_still_tol:
            _close_run(i)
            run_start_idx = i
            run_start_x, run_start_y = bx, by

    _close_run(n)
    return windows


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
    possessor_still_tol: float = 0.02,
    possessor_still_min_s: float = _DEFAULT_POSSESSOR_STILL_MIN_S,
    restart_stall_s: float = 15.0,
    use_referee_sidecar: bool = True,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
) -> list[StuckWindow]:
    """Slide a `window_s`-wide window (every `stride_s`) across a replay and
    flag windows where the ball is frozen (`ball_std < ball_still_tol`) and
    at least one friendly robot's position trace has non-DC spectral energy
    fraction above `oscillation_energy_tol` (`kind="oscillation"`), plus a
    second, independent pass (see "Restart stall" below) for a stalled
    restart ceremony that the oscillation signal can't see at all
    (`kind="restart_stall"`).

    Three false-positive classes are excluded before the frozen/oscillating
    check even runs, all found via a 2026-09-02 sweep of a competitive
    tournament re-run (every one of that run's "genuine" raw-flagged windows
    turned out to be one of these three, not a real stuck-match bug):

    - **Not live play.** A window where the referee command is
      `NORMAL_START`/`FORCE_START` for less than `live_play_fraction` of its
      frames is a legal stoppage/restart (kickoff standstill, a mid-match
      restart after a goal, a `STOP`<->`FORCE_START` violation cycle), not a
      stuck match — the ball is *supposed* to be frozen there. A replay
      whose frames carry no referee data (`frame.referee is None`) falls
      back, when `use_referee_sidecar` is set (the default), to the
      `<name>.intentions.jsonl` sidecar written alongside the replay (see
      `RefereeSidecar`) to reconstruct the command per frame; only if
      *that* is also unavailable does this check skip entirely, exactly as
      before — not every replay (a unit-test fixture, a non-refereed
      `debug_match.py` run) carries referee state or a sidecar.
    - **Ball possessed.** A window where some robot (either team) has
      `has_ball=True` for at least `possession_fraction` of its frames is a
      robot legitimately holding/shielding the ball, not a stall — this is
      the majority case in practice (e.g. a carrier paused mid-decision).
      This exclusion does *not* apply when a single robot possesses the
      ball continuously (at least `_CONTINUOUS_POSSESSION_FRACTION`, 95%,
      of the window — deliberately far stricter than `possession_fraction`
      itself) and is essentially motionless throughout (position std below
      `possessor_still_tol`) — a carrier parked on a frozen ball for
      basically the entire window, with nothing else resolving it, is the
      stall this detector exists to catch, not the legitimate "paused
      mid-decision while teammates/opponents jostle around it" case the
      exclusion was written for. The continuous-possession bar matters: an
      early version of this check used `possession_fraction` itself (0.3)
      and reintroduced exactly the `held_or_contested` false-positive
      class the possession exclusion was added to fix on 2026-09-02 — see
      `_CONTINUOUS_POSSESSION_FRACTION`'s docstring for the concrete
      `clear_danger_vs_high_press` case that caught it (an enemy robot
      held the ball motionless for 87% of a window while two unrelated
      friendly robots genuinely converged on it — real contested play, not
      a stall). The bypass also requires the possessor's *continuous* hold
      (tracked across the whole replay, not just one window — see
      `_possession_runs`) to span at least `possessor_still_min_s` seconds
      — a window-local-only version of this check flagged ordinary
      few-second ball-control moments (a kickoff-taker settling the ball
      before a pass) as stalls; see `_DEFAULT_POSSESSOR_STILL_MIN_S`'s
      docstring for the calibration sweep that found this and the resulting
      threshold.
    - **Ball in a defense area.** A window where the ball spends at least
      `defense_area_fraction` of its frames inside either team's defense
      box is already governed by the referee's own held-ball/interference
      rules (which resolve it via `STOP`/`BALL_PLACEMENT` on their own
      timeout, distinct from and faster than this detector's), so flagging
      it here would just be re-reporting a case the referee already handles.

    **Restart stall.** Independently of the sliding-window sweep above, a
    single linear pass looks for a maximal run of consecutive frames whose
    effective referee command (real or sidecar-reconstructed) is a
    *restart* command — anything other than `NORMAL_START`/`FORCE_START`
    (live play) or `HALT` (a deliberate freeze, never a stall) — while the
    ball stays within `ball_still_tol` of where the run started. A run
    lasting more than `restart_stall_s` (default 15s) is flagged
    `kind="restart_stall"`: no legal restart ceremony (a free-kick taker
    walking to the ball, a kickoff, a ball placement) should ever take that
    long, so a command stuck this way with the ball never moving is a stall
    the referee never auto-advanced out of — not a legal stoppage, and the
    "not live play" exclusion above must not hide it (it only excludes
    windows checked against `_LIVE_PLAY_COMMANDS`, which correctly treats
    *any* non-live command, restart or not, as "not live play"; this
    separate pass is what actually looks at how long that non-live state
    persisted). This class has no oscillation signal at all — a free-kick
    taker crawling toward the ball at well below any oscillation-detectable
    rate is still "not converging" in the sense that matters, so this pass
    doesn't require or check for oscillation; `oscillating_robot_ids` is
    always empty on a `restart_stall` window.

    Adjacent/overlapping flagged windows are merged before returning
    (separately within each `kind`, since an oscillation window and a
    restart-stall window covering the same span are two distinct findings,
    not one), and merged spans shorter than `min_duration_s` are dropped
    for `kind="oscillation"` — a single flagged 3s window on its own is
    exactly `window_s`, so `min_duration_s` only starts filtering once
    windows are tuned to overlap more (smaller `stride_s`) or `window_s`
    itself is shortened. `kind="restart_stall"` windows are already
    filtered by `restart_stall_s` and are not additionally subject to
    `min_duration_s`.

    Dispatches on `replay_path`'s extension: a `.npz` (columnar) replay is
    swept via `_find_stuck_windows_columnar`, working directly on the
    loaded numpy arrays with no per-tick `GameFrame` reconstruction at all
    (measured ~9ms for a full 600s match's ball-frozen pass alone, versus
    multi-second cost through object reconstruction) — see
    `columnar_writer.py`'s module docstring for why this format exists.
    Everything else falls through to the original `.pkl` path below.
    """
    replay_path = Path(replay_path)
    sidecar = RefereeSidecar.for_replay(replay_path) if use_referee_sidecar else RefereeSidecar(np.array([]), [])

    if replay_path.suffix == ".npz":
        return _find_stuck_windows_columnar(
            load_columnar_replay(replay_path),
            window_s=window_s,
            stride_s=stride_s,
            ball_still_tol=ball_still_tol,
            oscillation_energy_tol=oscillation_energy_tol,
            min_duration_s=min_duration_s,
            live_play_fraction=live_play_fraction,
            possession_fraction=possession_fraction,
            defense_area_fraction=defense_area_fraction,
            possessor_still_tol=possessor_still_tol,
            possessor_still_min_s=possessor_still_min_s,
            restart_stall_s=restart_stall_s,
            sidecar=sidecar,
            field_dims=field_dims,
        )

    frames: list[GameFrame] = [obj for obj in _load_replay(replay_path) if isinstance(obj, GameFrame)]
    if not frames:
        return []

    t0 = frames[0].ts
    t_last = frames[-1].ts

    # Effective per-frame referee command: the frame's own `referee` field
    # when present, otherwise the sidecar's reconstruction (or `None` if
    # neither is available) — computed once up front so both the
    # sliding-window sweep and the restart-stall pass below share it.
    effective_command: list[Optional[RefereeCommand]] = [
        f.referee.referee_command if f.referee is not None else sidecar.command_at(f.ts) for f in frames
    ]

    ts_arr = np.array([f.ts for f in frames])
    possessor_at: list[Optional[tuple[str, int]]] = []
    for f in frames:
        holders = [
            (team, rid)
            for team, robots in (("friendly", f.friendly_robots), ("enemy", f.enemy_robots))
            for rid, r in robots.items()
            if r.has_ball
        ]
        possessor_at.append(holders[0] if len(holders) == 1 else None)
    possession_runs = _possession_runs(possessor_at, ts_arr)

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
            if held_frac >= possession_fraction and not _possessor_is_motionless(
                window_frames,
                possessor_still_tol,
                window_start_idx=start_idx,
                window_end_idx=end_idx,
                possession_runs=possession_runs,
                possessor_still_min_s=possessor_still_min_s,
                ts=ts_arr,
            ):
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
                            kind="oscillation",
                        )
                    )
        t += stride_s

    ball_xs_all = np.array([f.ball.p.x if f.ball is not None else np.nan for f in frames])
    ball_ys_all = np.array([f.ball.p.y if f.ball is not None else np.nan for f in frames])
    restart_windows = _find_restart_stalls(
        ts=np.array([f.ts for f in frames]),
        ball_x=ball_xs_all,
        ball_y=ball_ys_all,
        commands=effective_command,
        ball_still_tol=ball_still_tol,
        restart_stall_s=restart_stall_s,
    )

    return _merge_windows(raw_windows, min_duration_s=min_duration_s) + restart_windows


def _find_stuck_windows_columnar(
    replay: ColumnarReplay,
    *,
    window_s: float,
    stride_s: float,
    ball_still_tol: float,
    oscillation_energy_tol: float,
    min_duration_s: float,
    live_play_fraction: float,
    possession_fraction: float,
    defense_area_fraction: float,
    possessor_still_tol: float,
    possessor_still_min_s: float,
    restart_stall_s: float,
    sidecar: RefereeSidecar,
    field_dims: FieldDimensions,
) -> list[StuckWindow]:
    """Array-native equivalent of the sliding-window sweep above. Every
    per-window reduction here (`live_frac`, `held_frac`, `defense_frac`,
    `ball_std`, oscillation energy) is the exact same formula as the
    `.pkl` path, just evaluated as a numpy slice-and-reduce instead of a
    Python loop over reconstructed `GameFrame`s — this is what turns a
    multi-second sweep into single-digit milliseconds (see this function's
    caller for the measured number). Output is the same `StuckWindow`
    sequence a `.pkl` version of the same replay would produce, modulo
    floating-point reduction order (immaterial at the tolerances used
    here).
    """
    n = replay.n_ticks
    if n == 0:
        return []

    ts = replay.ts
    t0 = float(ts[0])
    t_last = float(ts[-1])

    live_play_mask = np.isin(replay.referee_command, [c.value for c in _LIVE_PLAY_COMMANDS])
    held_mask = replay.friendly_has_ball.any(axis=1) | replay.enemy_has_ball.any(axis=1)
    half_length = field_dims.full_field_half_length
    depth = field_dims.half_defense_area_depth * 2
    half_width = field_dims.half_defense_area_width
    ball_x, ball_y = replay.ball_p[:, 0], replay.ball_p[:, 1]
    in_defense_mask = ((ball_x <= -half_length + depth) | (ball_x >= half_length - depth)) & (
        np.abs(ball_y) <= half_width
    )

    # Effective per-tick referee command, mirroring the `.pkl` path: the
    # tick's own command when `has_referee`, otherwise the sidecar's
    # reconstruction (or `None`) — shared by the restart-stall pass below.
    effective_command: list[Optional[RefereeCommand]] = []
    for i in range(n):
        if replay.has_referee[i]:
            effective_command.append(RefereeCommand.from_id(int(replay.referee_command[i])))
        else:
            effective_command.append(sidecar.command_at(float(ts[i])))

    # Per-tick sole possessor identity ((team, rid) or None if zero or more
    # than one robot holds the ball that tick), mirroring the `.pkl` path's
    # `possessor_at` — feeds `_possession_runs` so
    # `_possessor_is_motionless_columnar` can check a hold's *real* duration
    # across the whole replay, not just one window (see
    # `_DEFAULT_POSSESSOR_STILL_MIN_S`).
    n_held = replay.friendly_has_ball.sum(axis=1) + replay.enemy_has_ball.sum(axis=1)
    friendly_slot = np.argmax(replay.friendly_has_ball, axis=1)
    enemy_slot = np.argmax(replay.enemy_has_ball, axis=1)
    friendly_holds = replay.friendly_has_ball.any(axis=1)
    possessor_at: list[Optional[tuple[str, int]]] = []
    for i in range(n):
        if n_held[i] != 1:
            possessor_at.append(None)
        elif friendly_holds[i]:
            possessor_at.append(("friendly", int(replay.friendly_ids[friendly_slot[i]])))
        else:
            possessor_at.append(("enemy", int(replay.enemy_ids[enemy_slot[i]])))
    possession_runs = _possession_runs(possessor_at, ts)

    raw_windows: list[StuckWindow] = []
    start_idx = 0
    end_idx = 0
    t = t0
    while t + window_s <= t_last:
        while start_idx < n and ts[start_idx] < t:
            start_idx += 1
        if end_idx < start_idx:
            end_idx = start_idx
        while end_idx < n and ts[end_idx] <= t + window_s:
            end_idx += 1
        sl = slice(start_idx, end_idx)
        window_len = end_idx - start_idx

        if window_len >= 4 and not np.isnan(ball_x[sl]).any():
            has_referee_here = replay.has_referee[sl]
            if has_referee_here[0]:
                live_frac = live_play_mask[sl].sum() / window_len
                if live_frac < live_play_fraction:
                    t += stride_s
                    continue

            held_frac = held_mask[sl].sum() / window_len
            if held_frac >= possession_fraction and not _possessor_is_motionless_columnar(
                replay,
                sl,
                possessor_still_tol,
                possession_runs=possession_runs,
                possessor_still_min_s=possessor_still_min_s,
                ts=ts,
            ):
                t += stride_s
                continue

            defense_frac = in_defense_mask[sl].sum() / window_len
            if defense_frac >= defense_area_fraction:
                t += stride_s
                continue

            ball_std = float(np.hypot(ball_x[sl].std(), ball_y[sl].std()))

            if ball_std < ball_still_tol:
                # Only robots present (non-NaN) for the *entire* window
                # count, matching the `.pkl` path's
                # `set.intersection(*(set(f.friendly_robots) ...))`.
                present = ~np.isnan(replay.friendly_p[sl, :, 0]).any(axis=0)
                oscillating: list[int] = []
                for slot in np.flatnonzero(present):
                    xs = replay.friendly_p[sl, slot, 0]
                    ys = replay.friendly_p[sl, slot, 1]
                    energy = max(_dominant_non_dc_fraction(xs), _dominant_non_dc_fraction(ys))
                    if energy > oscillation_energy_tol:
                        oscillating.append(int(replay.friendly_ids[slot]))

                if oscillating:
                    raw_windows.append(
                        StuckWindow(
                            t_start=t,
                            t_end=t + window_s,
                            ball_std=ball_std,
                            oscillating_robot_ids=tuple(sorted(oscillating)),
                            kind="oscillation",
                        )
                    )
        t += stride_s

    restart_windows = _find_restart_stalls(
        ts=ts,
        ball_x=ball_x,
        ball_y=ball_y,
        commands=effective_command,
        ball_still_tol=ball_still_tol,
        restart_stall_s=restart_stall_s,
    )

    return _merge_windows(raw_windows, min_duration_s=min_duration_s) + restart_windows


def _possessor_is_motionless_columnar(
    replay: ColumnarReplay,
    sl: slice,
    possessor_still_tol: float,
    *,
    possession_runs: list[tuple[int, int, tuple[str, int]]],
    possessor_still_min_s: float,
    ts: np.ndarray,
) -> bool:
    """Array-native equivalent of `_possessor_is_motionless` — same
    semantics (a single robot holds the ball for at least
    `_CONTINUOUS_POSSESSION_FRACTION` of the window, is motionless for
    those frames, AND that robot's continuous hold spans at least
    `possessor_still_min_s` real seconds — see `_DEFAULT_POSSESSOR_STILL_MIN_S`),
    evaluated over `ColumnarReplay`'s arrays instead of a `GameFrame` list.
    """
    window_len = sl.stop - sl.start
    if window_len <= 0:
        return False
    for team, p, has_ball in (
        ("friendly", replay.friendly_p, replay.friendly_has_ball),
        ("enemy", replay.enemy_p, replay.enemy_has_ball),
    ):
        held = has_ball[sl]  # (window_len, n_robots)
        for slot in range(held.shape[1]):
            mask = held[:, slot]
            if mask.sum() / window_len < _CONTINUOUS_POSSESSION_FRACTION:
                continue
            xs = p[sl, slot, 0][mask]
            ys = p[sl, slot, 1][mask]
            std = float(np.hypot(np.std(xs), np.std(ys)))
            if std >= possessor_still_tol:
                continue
            rid = int(replay.friendly_ids[slot] if team == "friendly" else replay.enemy_ids[slot])
            key = (team, rid)
            if _run_duration_covering(possession_runs, key, sl.start, sl.stop, ts) >= possessor_still_min_s:
                return True
    return False


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
                kind=current.kind,
            )
        else:
            merged.append(current)
            current = w
    merged.append(current)

    return [w for w in merged if (w.t_end - w.t_start) >= min_duration_s]


def _scan_one(replay_path: Path) -> tuple[Path, list[StuckWindow]]:
    """Picklable top-level wrapper so `_scan_replays` can hand this to a
    `multiprocessing.Pool` (a bound method or local closure isn't
    picklable on the default `spawn`/`fork`-then-pickle start methods on
    every platform)."""
    return replay_path, find_stuck_windows(replay_path)


def _scan_replays(paths: list[Path], *, jobs: Optional[int] = None) -> list[tuple[Path, list[StuckWindow]]]:
    """Run `find_stuck_windows` over every path in `paths`, in parallel.
    Order of the returned list matches `paths`' input order regardless of
    which worker finishes first (`Pool.map` preserves input order), so CLI
    output is deterministic across runs. `jobs=1` runs serially in-process
    (useful for tests/debugging without paying multiprocessing's spawn
    cost) instead of forking a redundant single-worker pool.
    """
    if jobs == 1 or len(paths) <= 1:
        return [_scan_one(p) for p in paths]

    import multiprocessing

    with multiprocessing.Pool(processes=jobs) as pool:
        return pool.map(_scan_one, paths)


def _replays_in(target: Path) -> list[Path]:
    """A single replay file, or every `.pkl`/`.npz` replay directly inside
    a tournament directory (not recursive — a tournament dir is flat), in
    a stable sorted order."""
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix in (".pkl", ".npz"))


def _format_window(w: StuckWindow) -> str:
    span = f"{w.t_start:.2f}-{w.t_end:.2f}s"
    if w.kind == "restart_stall":
        return f"{w.kind}@{span} (ball_std={w.ball_std:.4f})"
    robots = ",".join(str(r) for r in w.oscillating_robot_ids)
    return f"{w.kind}@{span} (ball_std={w.ball_std:.4f}, robots=[{robots}])"


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Scan a replay (or every replay in a tournament directory) for "
            "stuck windows (see this module's docstring)."
        )
    )
    parser.add_argument("target", type=Path, help="A single .pkl/.npz replay, or a tournament directory of them.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text lines.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker processes for a directory scan (default: os.cpu_count()). Pass 1 to run serially.",
    )
    args = parser.parse_args()

    paths = _replays_in(args.target)
    if not paths:
        print(f"No .pkl/.npz replays found at {args.target}", file=__import__("sys").stderr)
        return 1

    results = _scan_replays(paths, jobs=args.jobs)
    flagged = [(p, ws) for p, ws in results if ws]

    if args.json:
        import json as _json

        payload = [
            {
                "replay": str(p),
                "windows": [
                    {
                        "kind": w.kind,
                        "t_start": w.t_start,
                        "t_end": w.t_end,
                        "ball_std": w.ball_std,
                        "oscillating_robot_ids": list(w.oscillating_robot_ids),
                    }
                    for w in ws
                ],
            }
            for p, ws in flagged
        ]
        print(_json.dumps({"flagged": payload, "n_scanned": len(paths), "n_flagged": len(flagged)}, indent=2))
    else:
        for p, ws in flagged:
            windows_str = "; ".join(_format_window(w) for w in ws)
            print(f"{p.name}: {windows_str}")
        print(f"Scanned {len(paths)} replay(s), {len(flagged)} flagged.")

    return 1 if flagged else 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
