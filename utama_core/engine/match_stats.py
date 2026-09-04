"""`MatchStats` — aggregate per-match summary, for post-match analysis.

Complements `utama_core.engine.match_log.MatchLog` (the "why" trace of
tactic decisions) with the "what happened" boxscore: rule-event counts
(goals, out-of-bounds, ...), ball-possession share, and pitch zone-time —
one small JSON object per match instead of a per-tick series, so it's cheap
to read across a whole tournament.

`GameHistory` (see `utama_core.entities.game.game_history`) is not used as
the data source here: it's bounded by `MAX_GAME_HISTORY` (20 frames), far
shorter than a full match, so possession/zone-time are accumulated live,
one tick at a time, via `record_tick()` rather than derived post-hoc from a
retained buffer.

`record_tick()` also runs a small in-match stall watchdog (`StallEvent`,
`MatchStats.stall_events`) alongside the boxscore accounting: RESTART_STALL
(a referee restart command that never auto-advances back to live play) and
COMMITTED_FROZEN (the ball not moving while a tactic slot stays committed).
Both are pure observations recorded for post-match reporting (see
`tournament.py`'s "STALLS" section) — nothing here reads back into or
alters gameplay.

`turnovers`/`completed_passes`/`attacking_third_entries` (friendly-side) and
their `enemy_*` counterparts reproduce `tools/metric_correlation.py`'s
offline definitions of the same names live, per tick, instead of requiring a
separate offline replay pass — see that module's docstring (metrics 2/3) for
the study that picked these three as correlated-with-outcome and run-to-run
reliable. The only definitional difference: the offline tool samples replay
frames at 10 Hz; `record_tick()` runs at rsim's full tick rate, which only
makes possession-state and boundary-crossing transitions *more* likely to be
observed, not less — see `_update_possession_events`'s docstring for detail.

`possession_under_pressure_s` and `restart_to_first_entry_s`/`n_restarts`
(both sides) are the same live port of that module's metrics 4/5 — see
`record_tick`'s "opponent counterparts" section and
`_maybe_resolve_restart_to_entry` for the live equivalents.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# Pitch is bucketed into thirds along x, from the recording side's own goal
# (defensive) to the opponent's goal (attacking), independent of which
# physical side ("left"/"right") the team is currently defending.
_ZONES = ("defensive", "mid", "attacking")

# Referee commands where play is actually live -- everything else (a
# restart ceremony, a stoppage) is expected to hold the ball/robots still,
# so the stall watchdog only measures "frozen ball" against these.
_LIVE_PLAY_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})

# RESTART_STALL: a non-live referee command (a restart/stoppage) that has
# held continuously for longer than this many sim seconds without
# auto-advancing back to a live command. A real kickoff/free-kick/ball-
# placement ceremony resolves in a few seconds; this is a generous margin
# above that, not a tight tolerance -- see docs/STRATEGY_DEVELOPMENT.md's
# Observability section for the matches this was built to catch.
_RESTART_STALL_SECONDS = 15.0
# COMMITTED_FROZEN: the ball has moved less than this many metres...
_STALL_BALL_STILL_TOL_M = 0.05
# ...for longer than this many sim seconds during live play, while at least
# one tactic slot is committed (or, if slot-commitment info isn't supplied,
# simply "during live play" -- see `record_tick`'s `committed_tactics` arg).
_COMMITTED_FROZEN_SECONDS = 10.0


@dataclass
class StallEvent:
    """One detected in-match stall -- an observation for post-match review,
    never fed back into gameplay (see `MatchStatsAccumulator._maybe_*` below,
    which only ever read referee/ball state, never write it).

    `sim_time`/`tick` are the *onset* of the stall (first tick past the
    threshold), not every tick it persisted -- `duration_s` is updated in
    place as the same stall continues, so one stall produces one event, not
    one per tick.
    """

    kind: str  # "RESTART_STALL" | "COMMITTED_FROZEN"
    sim_time: float
    tick: int
    referee_command: str
    duration_s: float
    tactic_ids: tuple = ()
    robot_ids: tuple = ()


@dataclass
class MatchStats:
    rule_event_counts: Dict[str, int]
    possession_pct: Dict[str, float]
    # Keys are "<side>_<robot_id>" (e.g. "friendly_3", "enemy_1") — robot ids
    # collide across teams in PVP frames, so the side prefix disambiguates.
    zone_time_pct: Dict[str, Dict[str, float]]
    shots: Dict[str, int] = field(default_factory=lambda: {"friendly": 0, "enemy": 0})
    ball_travel_m: float = 0.0
    robot_motion_pct: Dict[str, float] = field(default_factory=dict)
    stall_events: List[StallEvent] = field(default_factory=list)
    # Friendly-side counts, matching `tools/metric_correlation.py`'s
    # `*_friendly` values (config_a's perspective) -- see that module's
    # metric 2/3 definitions and this file's `_update_possession_events`/
    # attacking-third-entry block in `record_tick`.
    turnovers: int = 0
    completed_passes: int = 0
    attacking_third_entries: int = 0
    # Opponent counterparts to the three friendly-only fields above, plus
    # possession-under-pressure and restart-to-entry latency (metrics 4/5 in
    # `tools/metric_correlation.py`, ported live -- see that module's
    # docstring for the full definitions and `record_tick`'s "opponent
    # counterparts" section below for the live equivalents). All keyed
    # "friendly"/"enemy" like `possession_pct`/`shots`, so callers that
    # already read those don't need a new convention.
    enemy_turnovers: int = 0
    enemy_completed_passes: int = 0
    enemy_attacking_third_entries: int = 0
    possession_under_pressure_s: Dict[str, float] = field(default_factory=lambda: {"friendly": 0.0, "enemy": 0.0})
    # Mean seconds from a live-play restart (NORMAL_START/FORCE_START,
    # transitioning from a non-live command) to that restart's possessing
    # side's first entry into ITS OWN attacking third -- `None` per side if
    # that side never had a restart resolve to an entry this match (excluded
    # from the mean, not imputed -- see `tools/metric_correlation.py` metric
    # 5's docstring). `n_restarts`/`n_restarts_with_entry` are match-level
    # (both sides combined), matching the offline tool's fields of the same
    # name.
    restart_to_first_entry_s: Dict[str, Optional[float]] = field(
        default_factory=lambda: {"friendly": None, "enemy": None}
    )
    n_restarts: int = 0
    n_restarts_with_entry: int = 0

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "rule_event_counts": self.rule_event_counts,
                    "possession_pct": self.possession_pct,
                    "zone_time_pct": self.zone_time_pct,
                    "shots": self.shots,
                    "ball_travel_m": self.ball_travel_m,
                    "robot_motion_pct": self.robot_motion_pct,
                    "turnovers": self.turnovers,
                    "completed_passes": self.completed_passes,
                    "attacking_third_entries": self.attacking_third_entries,
                    "enemy_turnovers": self.enemy_turnovers,
                    "enemy_completed_passes": self.enemy_completed_passes,
                    "enemy_attacking_third_entries": self.enemy_attacking_third_entries,
                    "possession_under_pressure_s": self.possession_under_pressure_s,
                    "restart_to_first_entry_s": self.restart_to_first_entry_s,
                    "n_restarts": self.n_restarts,
                    "n_restarts_with_entry": self.n_restarts_with_entry,
                    "stall_events": [
                        {
                            "kind": e.kind,
                            "sim_time": e.sim_time,
                            "tick": e.tick,
                            "referee_command": e.referee_command,
                            "duration_s": e.duration_s,
                            "tactic_ids": list(e.tactic_ids),
                            "robot_ids": list(e.robot_ids),
                        }
                        for e in self.stall_events
                    ],
                },
                f,
                indent=2,
            )


# A "shot" is a box-score heuristic, not a referee event: a hard ball
# (>= this ground speed) moving toward the side's attacking goal from the
# attacking half. Puck/placement teleports and soft passes do not count.
_SHOT_SPEED_MPS = 3.5
# Ball slower than this releases the per-side shot lock (the same play's
# remaining ticks must not be recounted).
_SHOT_LOCK_RELEASE_MPS = 1.0
# Ball deltas larger than this are placements/teleports, not travel.
_PLACEMENT_JUMP_M = 2.0
# A straight-line extrapolation from this far out is unreliable as an
# on-target check (see the shots detector's docstring below): require the
# ball to already be in the attacking third, not just past midfield.
# Matches the `signed_x > 1.5` boundary `record_tick()`'s zone-time bucketing
# already uses for "attacking" -- one convention for "close enough to goal
# to matter", not two.
_SHOT_ATTACKING_THIRD_M = 1.5
# A robot moving faster than this is "in motion" for the motion-share stat.
_MOTION_SPEED_MPS = 0.15

# turnovers / completed_passes / attacking_third_entries -- live equivalents of
# tools/metric_correlation.py's offline definitions (`compute_frame_metrics`),
# same thresholds, reproduced here so every match records them without an
# offline replay pass. See `MatchStatsAccumulator._update_possession_events`
# and the `attacking_third_entries` block in `record_tick` for the exact
# state machines; the differences from the offline tool are documented there
# and in this module's own docstring update below.
#
# Possession radius: robot footprint + dribble slack (matches
# tools/metric_correlation.py's `_POSSESSION_RADIUS_M`).
_POSSESSION_RADIUS_M = 2 * ROBOT_RADIUS + 0.05
# Ball must be slower than this to count as "controlled" (matches
# tools/metric_correlation.py's `_PASS_RELEASE_MPS`, which itself matches
# this file's own `_SHOT_LOCK_RELEASE_MPS`).
_PASS_RELEASE_MPS = 1.0
# Hysteresis band around the attacking-third boundary (half_length +
# _SHOT_ATTACKING_THIRD_M) so oscillation on that line doesn't double-count
# entries. Matches tools/metric_correlation.py's `_ENTRY_HYSTERESIS_M`.
_ENTRY_HYSTERESIS_M = 0.3
# possession_under_pressure_s: an opponent robot within this range of the
# ball, while its own side possesses (nearest-robot-to-ball), counts as
# "under pressure" this tick. Matches tools/metric_correlation.py's
# `_PRESSURE_RADIUS_M` (metric 4).
_PRESSURE_RADIUS_M = 0.5


@dataclass
class MatchStatsAccumulator:
    """Accumulates per-tick possession/zone data and rule-violation counts; call `finalize()` once."""

    _rule_event_counts: Dict[str, int] = field(default_factory=dict)
    _possession_ticks: Dict[str, int] = field(default_factory=lambda: {"friendly": 0, "enemy": 0})
    _zone_ticks: Dict[int, Dict[str, int]] = field(default_factory=dict)
    _ticks_recorded: int = 0
    _shots: Dict[str, int] = field(default_factory=lambda: {"friendly": 0, "enemy": 0})
    _shot_lock: Dict[str, bool] = field(default_factory=lambda: {"friendly": False, "enemy": False})
    _last_ball_xy: Optional[tuple[float, float]] = None
    _ball_travel_m: float = 0.0
    _motion_ticks: Dict[int, int] = field(default_factory=dict)
    _measured_ticks: Dict[int, int] = field(default_factory=dict)

    # --- turnovers / completed_passes state (possession-radius state
    # machine, see `_update_possession_events`) ---
    # Side of the robot currently "holding" the ball (controlled: within
    # `_POSSESSION_RADIUS_M`, ball slower than `_PASS_RELEASE_MPS`), or the
    # side that last released it (robot_id cleared) while its outcome is
    # still unresolved, or (None, None) when no side has been on the ball
    # yet. Mirrors `tools/metric_correlation.py`'s `_PossessionState`.
    _poss_side: Optional[str] = None
    _poss_robot_id: Optional[int] = None
    _turnovers: int = 0
    _completed_passes: int = 0
    _enemy_turnovers: int = 0
    _enemy_completed_passes: int = 0

    # --- attacking_third_entries state (hysteresis, both sides -- friendly
    # feeds `attacking_third_entries`, enemy feeds `enemy_attacking_third_entries`
    # and both feed restart-to-entry resolution below) ---
    _in_attacking_third: Dict[str, bool] = field(default_factory=lambda: {"friendly": False, "enemy": False})
    _attacking_third_entries: int = 0
    _enemy_attacking_third_entries: int = 0

    # --- possession_under_pressure_s state (metric 4) ---
    # Accumulated directly in sim-seconds (consecutive `game_frame.ts`
    # deltas), not a tick count times an assumed rate -- `record_tick` never
    # sees rsim's configured step rate itself, and `sim_time` is already the
    # authoritative clock the stall watchdog uses for the same reason (see
    # `_maybe_record_stalls`'s `elapsed = sim_time - self._restart_started_at`).
    _pressure_s: Dict[str, float] = field(default_factory=lambda: {"friendly": 0.0, "enemy": 0.0})
    _last_tick_ts: Optional[float] = None

    # --- restart_to_first_entry_s state (metric 5): a live-play command
    # (NORMAL_START/FORCE_START) transitioning from a non-live command starts
    # a restart clock, attributed to whichever side is nearest the ball at
    # that instant; it resolves (and stops) at that same side's first
    # subsequent entry into its own attacking third (`_in_attacking_third`
    # above going True). Mirrors `tools/metric_correlation.py`'s
    # `pending_restart`/`restarts_resolved` exactly, driven off
    # `record_tick`'s own live-command-transition tracking
    # (`_restart_command`, already needed by the RESTART_STALL watchdog)
    # instead of a second referee-event pass over intentions.jsonl.
    _pending_restart_t0: Optional[float] = None
    _pending_restart_side: Optional[str] = None
    _restart_entry_times: Dict[str, List[float]] = field(default_factory=lambda: {"friendly": [], "enemy": []})
    _n_restarts: int = 0
    _n_restarts_with_entry: int = 0

    # --- stall watchdog state (see `_maybe_record_restart_stall`/
    # `_maybe_record_committed_frozen` below) ---
    _stall_events: List[StallEvent] = field(default_factory=list)
    # Sim time the current non-live referee command started, and which
    # command that is -- reset on every command change.
    _restart_command: Optional[RefereeCommand] = None
    _restart_started_at: Optional[float] = None
    # Set once a RESTART_STALL has already been logged for the *current*
    # restart, so a 65s stuck restart produces one event, not one per tick
    # past the threshold.
    _restart_stall_logged: bool = False
    # Index into `_stall_events` of the in-progress RESTART_STALL, so its
    # `duration_s` can keep being updated while the same restart persists.
    _restart_stall_event_idx: Optional[int] = None

    # Sim time + position the ball was last seen having moved
    # >= _STALL_BALL_STILL_TOL_M from -- reset whenever it moves that far.
    _ball_still_since: Optional[float] = None
    _ball_still_anchor_xy: Optional[tuple[float, float]] = None
    _committed_frozen_logged: bool = False
    _committed_frozen_event_idx: Optional[int] = None

    def record_rule_violation(self, violation: Optional[RuleViolation]) -> None:
        if violation is None:
            return
        self._rule_event_counts[violation.rule_name] = self._rule_event_counts.get(violation.rule_name, 0) + 1

    def record_tick(
        self,
        game_frame: GameFrame,
        committed_tactics: Optional[Dict[str, tuple]] = None,
    ) -> None:
        """Attribute possession and zone occupancy for one tick's `GameFrame`.

        `committed_tactics`, if given, is the current tick's committed-slot
        info as `{tactic_id: (robot_id, ...)}` -- cheaply available from the
        kernel `Strategy` via `AbstractStrategy.debug_status()` (already
        computed every tick for the referee debug GUI panel; see
        `StrategyRunner._push_bt_nodes_to_referee`), so this accumulator
        doesn't need to reach into kernel internals itself. When omitted
        (`None`, the default -- e.g. a BT-path strategy, or a caller that
        doesn't have it handy), `COMMITTED_FROZEN` falls back to "ball frozen
        during live play", without requiring a committed slot; the resulting
        `StallEvent.tactic_ids`/`robot_ids` are then simply empty.
        """
        ball = game_frame.ball
        all_robots = (
            [(rid, robot, "friendly") for rid, robot in game_frame.friendly_robots.items()]
            + [(rid, robot, "enemy") for rid, robot in game_frame.enemy_robots.items()]
            if ball is not None
            else []
        )
        # Must run before `_maybe_record_stalls` mutates `_restart_command`
        # to the current tick's value -- this needs to see the *previous*
        # tick's command to detect the same transition itself (see that
        # method's own restart-command tracking, reused here rather than
        # duplicated: `_maybe_start_restart_clock` does its own comparison
        # against `_restart_command` a few lines before `_maybe_record_stalls`
        # would otherwise overwrite it).
        self._maybe_start_restart_clock(game_frame, all_robots, ball)
        self._maybe_record_stalls(game_frame, committed_tactics)

        if ball is None or not all_robots:
            return

        self._ticks_recorded += 1

        # Ball travel, skipping placement/teleport jumps.
        ball_xy = (ball.p.x, ball.p.y)
        if self._last_ball_xy is not None:
            delta = math.hypot(ball_xy[0] - self._last_ball_xy[0], ball_xy[1] - self._last_ball_xy[1])
            if delta < _PLACEMENT_JUMP_M:
                self._ball_travel_m += delta
        self._last_ball_xy = ball_xy

        # Shot attempts: hard balls hit toward the attacking goal from the
        # attacking third, on a straight-line trajectory that actually reaches
        # the goal mouth (not just "moving in the right x-direction with some
        # speed", which also passes for a hard clear or cross-field switch —
        # see `docs/roadmap.md`'s "why is it 0-0" investigation, which found
        # this heuristic counting balls whose extrapolated path missed the
        # goal by several metres of lateral distance). "Attacking third", not
        # merely "attacking half" (found live in a 2026-09-02 tournament
        # re-run: a center-circle clearance at (-0.08, -0.09) with speed
        # 3.83 m/s satisfied the old attacking-half gate and, purely by
        # chance, extrapolated to land inside the 1m-wide goal mouth 4.5m
        # away — the ball's real velocity collapsed within 1-2 ticks in
        # every traced case, well before reaching goal, since a straight-
        # line projection from that far out ignores the interception/
        # friction/spin that make long-range "shots" essentially never
        # arrive as aimed). Edge-detected per side (locked until the ball
        # slows). `GameFrame` carries no field geometry, so boxscore
        # heuristics use the standard SSL field dims (the sim always plays
        # on them).
        half_length = STANDARD_FIELD_DIMS.full_field_half_length
        half_goal_width = STANDARD_FIELD_DIMS.half_goal_width
        own_goal_sign = 1.0 if game_frame.my_team_is_right else -1.0
        for side, attack_sign in (("friendly", -own_goal_sign), ("enemy", own_goal_sign)):
            speed = math.hypot(ball.v.x, ball.v.y)
            if speed < _SHOT_LOCK_RELEASE_MPS:
                self._shot_lock[side] = False
                continue
            if self._shot_lock[side]:
                continue
            # Progress is measured from each side's *own* goal line (their
            # defensive line), so ``> half_length + _SHOT_ATTACKING_THIRD_M``
            # is precisely "ball in that side's attacking third", not merely
            # past midfield.
            ref_goal_x = (own_goal_sign if side == "friendly" else -own_goal_sign) * half_length
            progress_from_own_goal = (ball.p.x - ref_goal_x) * attack_sign
            toward_goal = ball.v.x * attack_sign > 0.0
            if not (
                speed >= _SHOT_SPEED_MPS
                and toward_goal
                and progress_from_own_goal > half_length + _SHOT_ATTACKING_THIRD_M
            ):
                continue
            # On-target check: extrapolate the current straight-line velocity
            # to the attacking goal's line (x = attack_sign * half_length)
            # and require the predicted y to land within the goal mouth.
            # `ball.v.x` can't be ~0 here (`toward_goal` + `speed >=
            # _SHOT_SPEED_MPS` already bound it away from 0), so this
            # division is safe.
            attacking_goal_x = attack_sign * half_length
            ticks_to_goal_line = (attacking_goal_x - ball.p.x) / ball.v.x
            predicted_y_at_goal = ball.p.y + ball.v.y * ticks_to_goal_line
            if abs(predicted_y_at_goal) <= half_goal_width:
                self._shots[side] += 1
                self._shot_lock[side] = True

        nearest_id, nearest_robot, nearest_side = min(all_robots, key=lambda entry: entry[1].p.distance_to(ball.p))
        dist_to_nearest = nearest_robot.p.distance_to(ball.p)
        self._possession_ticks[nearest_side] += 1
        self._update_possession_events(nearest_side, nearest_id, dist_to_nearest, ball)

        # possession_under_pressure_s (metric 4): the possessing side's
        # nearest-to-ball robot also has an opponent within
        # `_PRESSURE_RADIUS_M`. Accumulated in real sim-seconds via the
        # elapsed `game_frame.ts` since the last recorded tick (see
        # `_pressure_s`'s field docstring) -- same "skip an anomalous jump"
        # guard `_ball_travel_m` uses above, since a placement/reset can
        # otherwise make `ts` jump backward or by an outsized amount.
        sim_ts = game_frame.ts
        dt = 0.0
        if self._last_tick_ts is not None:
            candidate_dt = sim_ts - self._last_tick_ts
            if 0.0 < candidate_dt < 1.0:
                dt = candidate_dt
        self._last_tick_ts = sim_ts
        opp_side = "enemy" if nearest_side == "friendly" else "friendly"
        opponents = [r for rid, r, s in all_robots if s == opp_side]
        if dist_to_nearest <= _POSSESSION_RADIUS_M and opponents:
            min_opp_dist = min(r.p.distance_to(ball.p) for r in opponents)
            if min_opp_dist <= _PRESSURE_RADIUS_M:
                self._pressure_s[nearest_side] += dt

        for rid, robot, side in all_robots:
            # Motion share: what fraction of measured ticks a robot actually moved.
            key = f"{side}_{rid}"  # robot ids collide across teams in PVP frames
            if robot.v is not None:
                self._measured_ticks[key] = self._measured_ticks.get(key, 0) + 1
                if math.hypot(robot.v.x, robot.v.y) > _MOTION_SPEED_MPS:
                    self._motion_ticks[key] = self._motion_ticks.get(key, 0) + 1

        # "attacking" always means "towards the robot's own attacking goal"
        # regardless of which physical side (x > 0 vs x < 0) that currently
        # is. Matches the `own_goal_sign = 1.0 if my_team_is_right else -1.0`
        # convention in `strategy/referee/actions.py` — friendly's own goal
        # sits at `+own_goal_sign * half_length`, so friendly attacks toward
        # `-own_goal_sign`; enemy's own goal is the mirror, so enemy attacks
        # toward `+own_goal_sign`.
        own_goal_sign = 1.0 if game_frame.my_team_is_right else -1.0
        for rid, robot, side in all_robots:
            attack_sign = -own_goal_sign if side == "friendly" else own_goal_sign
            signed_x = robot.p.x * attack_sign
            if signed_x < -1.5:
                zone = "defensive"
            elif signed_x > 1.5:
                zone = "attacking"
            else:
                zone = "mid"
            key = f"{side}_{rid}"  # robot ids collide across teams in PVP frames
            zone_counts = self._zone_ticks.setdefault(key, {z: 0 for z in _ZONES})
            zone_counts[zone] += 1

        # attacking_third_entries (both sides): count of the *ball* crossing
        # from at/behind the entry threshold into a side's attacking third,
        # hysteresis-banded so oscillation on the boundary doesn't
        # double-count. Matches tools/metric_correlation.py's metric 2
        # exactly -- same threshold (half_length + _SHOT_ATTACKING_THIRD_M),
        # same hysteresis width.
        enter_threshold = half_length + _SHOT_ATTACKING_THIRD_M
        exit_threshold = enter_threshold - _ENTRY_HYSTERESIS_M
        for side in ("friendly", "enemy"):
            attack_sign = -own_goal_sign if side == "friendly" else own_goal_sign
            own_goal_x = own_goal_sign * half_length if side == "friendly" else -own_goal_sign * half_length
            progress = (ball.p.x - own_goal_x) * attack_sign
            if not self._in_attacking_third[side] and progress > enter_threshold:
                self._in_attacking_third[side] = True
                if side == "friendly":
                    self._attacking_third_entries += 1
                else:
                    self._enemy_attacking_third_entries += 1
                # restart_to_first_entry_s resolution (metric 5): this is
                # exactly the "possessing side reaches its own attacking
                # third" event `_maybe_start_restart_clock`'s pending clock
                # is waiting for. Matches tools/metric_correlation.py's
                # restart-resolution condition exactly (same side, same
                # attacking-third-entry event).
                if self._pending_restart_side == side and self._pending_restart_t0 is not None:
                    self._restart_entry_times[side].append(game_frame.ts - self._pending_restart_t0)
                    self._n_restarts_with_entry += 1
                    self._pending_restart_t0 = None
                    self._pending_restart_side = None
            elif self._in_attacking_third[side] and progress < exit_threshold:
                self._in_attacking_third[side] = False

    def _update_possession_events(self, nearest_side: str, nearest_id: int, dist_to_nearest: float, ball: Ball) -> None:
        """turnovers / completed_passes: a possession-radius state machine,
        reproducing `tools/metric_correlation.py`'s `compute_frame_metrics`
        metric 3 exactly (same `_POSSESSION_RADIUS_M`/`_PASS_RELEASE_MPS`
        thresholds), but driven every tick (this runs at rsim's full ~60 Hz
        step rate) rather than the offline tool's 10 Hz replay sample -- the
        offline tool trades tick resolution for replay-read cost, which
        doesn't apply here since this already runs inside the live tick
        loop. A possession-state transition (control gained/lost) is exactly
        as likely to be observed at 60 Hz as at 10 Hz or more so, so this is
        the closest live equivalent, not an approximation of it.

        Both sides' completions/turnovers are tallied (`_turnovers`/
        `_completed_passes` for friendly, `_enemy_turnovers`/
        `_enemy_completed_passes` for enemy) -- the state machine already
        tracks *either* side's possession (a turnover is defined by the ball
        moving from one side's holder to the other's), so attributing both
        outcomes is the same state, not new tracking. Matches
        `tools/metric_correlation.py`'s offline `completed_passes`/
        `turnovers` dicts, which are computed per-side there too.
        """
        ball_speed = math.hypot(ball.v.x, ball.v.y)
        controlled = dist_to_nearest <= _POSSESSION_RADIUS_M and ball_speed < _PASS_RELEASE_MPS
        if controlled:
            if self._poss_side is None:
                self._poss_side, self._poss_robot_id = nearest_side, nearest_id
            elif (self._poss_side, self._poss_robot_id) != (nearest_side, nearest_id):
                # Possession changed hands without an intervening "released
                # at speed" event (e.g. a slow dribble handoff/tackle) --
                # attribute as a same-side completed pass or a turnover
                # exactly like a released pass would, then adopt the new
                # holder.
                if self._poss_side == nearest_side == "friendly":
                    self._completed_passes += 1
                elif self._poss_side == nearest_side == "enemy":
                    self._enemy_completed_passes += 1
                elif self._poss_side == "friendly" and nearest_side != "friendly":
                    self._turnovers += 1
                elif self._poss_side == "enemy" and nearest_side != "enemy":
                    self._enemy_turnovers += 1
                self._poss_side, self._poss_robot_id = nearest_side, nearest_id
        elif self._poss_side is not None and ball_speed >= _PASS_RELEASE_MPS and dist_to_nearest > _POSSESSION_RADIUS_M:
            # Ball just left a controlled possession at speed: released. The
            # eventual outcome (pass vs turnover vs neither) is resolved the
            # next time the ball is controlled again, or never if it isn't.
            if self._poss_robot_id is not None:
                self._poss_robot_id = None

    def _maybe_start_restart_clock(
        self,
        game_frame: GameFrame,
        all_robots: List[tuple],
        ball: Optional[Ball],
    ) -> None:
        """restart_to_first_entry_s (metric 5) clock start: a live-play
        command (NORMAL_START/FORCE_START) just began from a non-live
        command. Attribute the restart to whichever side is nearest the
        ball this instant, matching `tools/metric_correlation.py`'s
        `pending_restart` exactly (same trigger condition, same
        nearest-robot-to-ball attribution). Resolution (stopping the clock)
        happens in `record_tick`'s attacking-third-entries block, since that
        is where each side's `_in_attacking_third` transition is already
        computed.

        Must run before `_maybe_record_stalls` overwrites `_restart_command`
        for this tick -- see `record_tick`'s call-order comment.
        """
        referee = game_frame.referee
        current_command = referee.referee_command if referee is not None else None
        was_live = self._restart_command in _LIVE_PLAY_COMMANDS
        is_live = current_command in _LIVE_PLAY_COMMANDS
        if is_live and not was_live and current_command != self._restart_command:
            self._n_restarts += 1
            nearest_side = (
                min(all_robots, key=lambda entry: entry[1].p.distance_to(ball.p))[2]
                if all_robots and ball is not None
                else None
            )
            self._pending_restart_t0 = game_frame.ts
            self._pending_restart_side = nearest_side

    def _maybe_record_stalls(self, game_frame: GameFrame, committed_tactics: Optional[Dict[str, tuple]]) -> None:
        """Update the two stall watchdogs for this tick. Pure observation --
        reads `game_frame`/`committed_tactics`, never mutates or influences
        gameplay (see `StallEvent`'s docstring).

        Both watchdogs are "first occurrence + running duration": each
        records its onset tick/sim_time once (`_restart_stall_logged`/
        `_committed_frozen_logged`), then keeps updating that same
        `StallEvent.duration_s` in place for as long as the same stall
        persists, rather than appending a new event every tick past the
        threshold.
        """
        referee = game_frame.referee
        sim_time = game_frame.ts
        tick = self._ticks_recorded + 1  # this tick hasn't incremented _ticks_recorded yet

        # --- RESTART_STALL: a non-live referee command held too long ---
        current_command = referee.referee_command if referee is not None else None
        if current_command != self._restart_command:
            self._restart_command = current_command
            self._restart_started_at = sim_time
            self._restart_stall_logged = False
            self._restart_stall_event_idx = None

        if current_command is not None and current_command not in _LIVE_PLAY_COMMANDS:
            elapsed = sim_time - self._restart_started_at
            if elapsed > _RESTART_STALL_SECONDS:
                if not self._restart_stall_logged:
                    self._restart_stall_logged = True
                    self._restart_stall_event_idx = len(self._stall_events)
                    self._stall_events.append(
                        StallEvent(
                            kind="RESTART_STALL",
                            sim_time=self._restart_started_at + _RESTART_STALL_SECONDS,
                            tick=tick,
                            referee_command=current_command.name,
                            duration_s=elapsed,
                        )
                    )
                elif self._restart_stall_event_idx is not None:
                    self._stall_events[self._restart_stall_event_idx].duration_s = elapsed

        # --- COMMITTED_FROZEN: ball frozen during live play while committed ---
        ball = game_frame.ball
        is_live = current_command in _LIVE_PLAY_COMMANDS
        if ball is None or not is_live:
            self._ball_still_since = None
            self._ball_still_anchor_xy = None
            self._committed_frozen_logged = False
            self._committed_frozen_event_idx = None
            return

        ball_xy = (ball.p.x, ball.p.y)
        if self._ball_still_anchor_xy is None or (
            math.hypot(ball_xy[0] - self._ball_still_anchor_xy[0], ball_xy[1] - self._ball_still_anchor_xy[1])
            >= _STALL_BALL_STILL_TOL_M
        ):
            self._ball_still_since = sim_time
            self._ball_still_anchor_xy = ball_xy
            self._committed_frozen_logged = False
            self._committed_frozen_event_idx = None
            return

        frozen_for = sim_time - self._ball_still_since
        if frozen_for <= _COMMITTED_FROZEN_SECONDS:
            return

        committed_tactic_ids: tuple = ()
        committed_robot_ids: tuple = ()
        if committed_tactics is not None:
            if not committed_tactics:
                # Slot-commitment info was supplied but nothing is committed
                # right now -- not the bug this watchdog targets (a carrier
                # holding forever, a handshake that never completes), so
                # don't flag it. This is the precise "at least one slot is
                # committed" gate from the spec.
                return
            committed_tactic_ids = tuple(sorted(committed_tactics.keys()))
            committed_robot_ids = tuple(sorted(rid for robots in committed_tactics.values() for rid in robots))
        # else: committed_tactics is None -- fall back to "ball frozen during
        # live play", per this method's docstring / `record_tick`'s.

        if not self._committed_frozen_logged:
            self._committed_frozen_logged = True
            self._committed_frozen_event_idx = len(self._stall_events)
            self._stall_events.append(
                StallEvent(
                    kind="COMMITTED_FROZEN",
                    sim_time=self._ball_still_since + _COMMITTED_FROZEN_SECONDS,
                    tick=tick,
                    referee_command=current_command.name if current_command is not None else "",
                    duration_s=frozen_for,
                    tactic_ids=committed_tactic_ids,
                    robot_ids=committed_robot_ids,
                )
            )
        elif self._committed_frozen_event_idx is not None:
            self._stall_events[self._committed_frozen_event_idx].duration_s = frozen_for

    def finalize(self) -> MatchStats:
        total = max(1, self._ticks_recorded)
        possession_pct = {side: count / total for side, count in self._possession_ticks.items()}
        zone_time_pct: Dict[str, Dict[str, float]] = {}
        for key, zone_counts in self._zone_ticks.items():
            zone_total = max(1, sum(zone_counts.values()))
            zone_time_pct[key] = {zone: count / zone_total for zone, count in zone_counts.items()}
        robot_motion_pct: Dict[str, float] = {}
        for key, measured in self._measured_ticks.items():
            robot_motion_pct[key] = self._motion_ticks.get(key, 0) / max(1, measured)
        restart_to_first_entry_s: Dict[str, Optional[float]] = {}
        for side in ("friendly", "enemy"):
            times = self._restart_entry_times[side]
            restart_to_first_entry_s[side] = round(sum(times) / len(times), 3) if times else None
        return MatchStats(
            rule_event_counts=dict(self._rule_event_counts),
            possession_pct=possession_pct,
            zone_time_pct=zone_time_pct,
            shots=dict(self._shots),
            ball_travel_m=round(self._ball_travel_m, 2),
            robot_motion_pct=robot_motion_pct,
            stall_events=list(self._stall_events),
            turnovers=self._turnovers,
            completed_passes=self._completed_passes,
            attacking_third_entries=self._attacking_third_entries,
            enemy_turnovers=self._enemy_turnovers,
            enemy_completed_passes=self._enemy_completed_passes,
            enemy_attacking_third_entries=self._enemy_attacking_third_entries,
            possession_under_pressure_s={side: round(s, 3) for side, s in self._pressure_s.items()},
            restart_to_first_entry_s=restart_to_first_entry_s,
            n_restarts=self._n_restarts,
            n_restarts_with_entry=self._n_restarts_with_entry,
        )
