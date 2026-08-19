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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

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
    zone_time_pct: Dict[int, Dict[str, float]]

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "rule_event_counts": self.rule_event_counts,
                    "possession_pct": self.possession_pct,
                    "zone_time_pct": self.zone_time_pct,
                },
                f,
                indent=2,
            )


@dataclass
class MatchStatsAccumulator:
    """Accumulates per-tick possession/zone data and rule-violation counts; call `finalize()` once."""

    _rule_event_counts: Dict[str, int] = field(default_factory=dict)
    _possession_ticks: Dict[str, int] = field(default_factory=lambda: {"friendly": 0, "enemy": 0})
    _zone_ticks: Dict[int, Dict[str, int]] = field(default_factory=dict)
    _ticks_recorded: int = 0

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

        nearest_side = min(all_robots, key=lambda entry: entry[1].p.distance_to(ball.p))[2]
        self._possession_ticks[nearest_side] += 1

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
            zone_counts = self._zone_ticks.setdefault(rid, {z: 0 for z in _ZONES})
            zone_counts[zone] += 1

    def finalize(self) -> MatchStats:
        total = max(1, self._ticks_recorded)
        possession_pct = {side: count / total for side, count in self._possession_ticks.items()}
        zone_time_pct: Dict[int, Dict[str, float]] = {}
        for rid, zone_counts in self._zone_ticks.items():
            zone_total = max(1, sum(zone_counts.values()))
            zone_time_pct[rid] = {zone: count / zone_total for zone, count in zone_counts.items()}
        return MatchStats(
            rule_event_counts=dict(self._rule_event_counts),
            possession_pct=possession_pct,
            zone_time_pct=zone_time_pct,
        )
