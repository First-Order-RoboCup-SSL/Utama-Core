"""Tests for `utama_core.kernel.match_stats` — the aggregate per-match boxscore."""

from __future__ import annotations

import json

from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel.match_stats import MatchStatsAccumulator


def _robot(rid: int, x: float, y: float, is_friendly: bool) -> Robot:
    return Robot(id=rid, is_friendly=is_friendly, has_ball=False, p=Vector2D(x, y), v=None, a=None, orientation=0)


def _frame(friendly, enemy, ball_xy, my_team_is_right: bool = True) -> GameFrame:
    ball = Ball(Vector3D(ball_xy[0], ball_xy[1], 0), Vector3D(0, 0, 0), None) if ball_xy is not None else None
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=ball,
    )


def _violation(rule_name: str) -> RuleViolation:
    return RuleViolation(
        rule_name=rule_name,
        suggested_command=RefereeCommand.STOP,
        next_command=None,
        status_message="",
    )


def test_rule_violation_tally_counts_by_name():
    acc = MatchStatsAccumulator()
    acc.record_rule_violation(_violation("goal"))
    acc.record_rule_violation(_violation("goal"))
    acc.record_rule_violation(_violation("out_of_bounds"))
    acc.record_rule_violation(None)  # no violation this tick — must not be tallied

    stats = acc.finalize()
    assert stats.rule_event_counts == {"goal": 2, "out_of_bounds": 1}


def test_possession_attributed_to_nearer_team():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 5.0, 0.0, False)}

    # Ball near friendly robot -> friendly possession this tick.
    acc.record_tick(_frame(friendly, enemy, ball_xy=(0.1, 0.0)))
    # Ball near enemy robot -> enemy possession this tick.
    acc.record_tick(_frame(friendly, enemy, ball_xy=(4.9, 0.0)))
    # Another friendly-possession tick.
    acc.record_tick(_frame(friendly, enemy, ball_xy=(0.0, 0.1)))

    stats = acc.finalize()
    assert stats.possession_pct["friendly"] == 2 / 3
    assert stats.possession_pct["enemy"] == 1 / 3


def test_zone_time_stationary_robot_is_all_one_zone():
    acc = MatchStatsAccumulator()
    # Friendly robot parked deep in its own attacking third (my_team_is_right=True
    # -> own goal at +half_length -> friendly attacks toward negative x).
    friendly = {1: _robot(1, -4.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}

    for _ in range(5):
        acc.record_tick(_frame(friendly, enemy, ball_xy=(0.0, 0.0)))

    stats = acc.finalize()
    assert stats.zone_time_pct[1]["attacking"] == 1.0
    assert stats.zone_time_pct[1]["defensive"] == 0.0
    assert stats.zone_time_pct[1]["mid"] == 0.0


def test_zone_time_midfield_robot_buckets_mid():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}
    acc.record_tick(_frame(friendly, enemy, ball_xy=(0.0, 0.0)))

    stats = acc.finalize()
    assert stats.zone_time_pct[1]["mid"] == 1.0


def test_no_ticks_recorded_yields_empty_stats_without_crashing():
    acc = MatchStatsAccumulator()
    stats = acc.finalize()
    assert stats.possession_pct == {"friendly": 0.0, "enemy": 0.0}
    assert stats.zone_time_pct == {}
    assert stats.rule_event_counts == {}


def test_to_json_round_trips(tmp_path):
    acc = MatchStatsAccumulator()
    acc.record_rule_violation(_violation("goal"))
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}
    acc.record_tick(_frame(friendly, enemy, ball_xy=(0.0, 0.0)))

    out = tmp_path / "stats.json"
    acc.finalize().to_json(out)

    data = json.loads(out.read_text())
    assert data["rule_event_counts"] == {"goal": 1}
    assert data["possession_pct"]["friendly"] == 1.0
    assert data["zone_time_pct"]["1"]["mid"] == 1.0
