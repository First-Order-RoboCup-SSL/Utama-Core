"""`MatchStats` — aggregate per-match summary, for post-match analysis.

Complements `utama_core.kernel.match_log.MatchLog` (the "why" trace of
tactic decisions) with the "what happened" boxscore: rule-event counts
(goals, out-of-bounds, ...), ball-possession share, and pitch zone-time —
one small JSON object per match instead of a per-tick series, so it's cheap
to read across a whole tournament.

`GameHistory` (see `utama_core.entities.game.game_history`) is not used as
the data source here: it's bounded by `MAX_GAME_HISTORY` (20 frames), far
shorter than a full match, so possession/zone-time are accumulated live,
one tick at a time, via `record_tick()` rather than derived post-hoc from a
retained buffer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.entities.game.game_frame import GameFrame

# Pitch is bucketed into thirds along x, from the recording side's own goal
# (defensive) to the opponent's goal (attacking), independent of which
# physical side ("left"/"right") the team is currently defending.
_ZONES = ("defensive", "mid", "attacking")


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
# A robot moving faster than this is "in motion" for the motion-share stat.
_MOTION_SPEED_MPS = 0.15


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

    def record_rule_violation(self, violation: Optional[RuleViolation]) -> None:
        if violation is None:
            return
        self._rule_event_counts[violation.rule_name] = self._rule_event_counts.get(violation.rule_name, 0) + 1

    def record_tick(self, game_frame: GameFrame) -> None:
        """Attribute possession and zone occupancy for one tick's `GameFrame`."""
        ball = game_frame.ball
        if ball is None:
            return

        all_robots = [(rid, robot, "friendly") for rid, robot in game_frame.friendly_robots.items()] + [
            (rid, robot, "enemy") for rid, robot in game_frame.enemy_robots.items()
        ]
        if not all_robots:
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
        # attacking half, edge-detected per side (locked until the ball slows).
        # `GameFrame` carries no field geometry, so boxscore heuristics use
        # the standard SSL field dims (the sim always plays on them).
        half_length = STANDARD_FIELD_DIMS.full_field_half_length
        own_goal_sign = 1.0 if game_frame.my_team_is_right else -1.0
        for side, attack_sign in (("friendly", -own_goal_sign), ("enemy", own_goal_sign)):
            speed = math.hypot(ball.v.x, ball.v.y)
            if speed < _SHOT_LOCK_RELEASE_MPS:
                self._shot_lock[side] = False
                continue
            if self._shot_lock[side]:
                continue
            # Progress is measured from each side's *own* goal line (their
            # defensive line), so ``> half_length`` is precisely \"ball past
            # midfield in that side's attacking half\".
            ref_goal_x = (own_goal_sign if side == "friendly" else -own_goal_sign) * half_length
            progress_from_own_goal = (ball.p.x - ref_goal_x) * attack_sign
            toward_goal = ball.v.x * attack_sign > 0.0
            if speed >= _SHOT_SPEED_MPS and toward_goal and progress_from_own_goal > half_length:
                self._shots[side] += 1
                self._shot_lock[side] = True

        nearest_side = min(all_robots, key=lambda entry: entry[1].p.distance_to(ball.p))[2]
        self._possession_ticks[nearest_side] += 1

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
        return MatchStats(
            rule_event_counts=dict(self._rule_event_counts),
            possession_pct=possession_pct,
            zone_time_pct=zone_time_pct,
            shots=dict(self._shots),
            ball_travel_m=round(self._ball_travel_m, 2),
            robot_motion_pct=robot_motion_pct,
        )
