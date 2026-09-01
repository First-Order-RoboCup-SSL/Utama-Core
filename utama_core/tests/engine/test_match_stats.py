"""Tests for `utama_core.engine.match_stats` — the aggregate per-match boxscore."""

from __future__ import annotations

import json

from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.engine.match_stats import MatchStatsAccumulator
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand


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
    assert stats.zone_time_pct["friendly_1"]["attacking"] == 1.0
    assert stats.zone_time_pct["friendly_1"]["defensive"] == 0.0
    assert stats.zone_time_pct["friendly_1"]["mid"] == 0.0


def test_zone_time_midfield_robot_buckets_mid():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}
    acc.record_tick(_frame(friendly, enemy, ball_xy=(0.0, 0.0)))

    stats = acc.finalize()
    assert stats.zone_time_pct["friendly_1"]["mid"] == 1.0


def test_robot_ids_kepied_per_side_across_teams():
    """Same robot id on both teams must not collide in zone/motion keys."""
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {1: _robot(1, 4.0, 0.0, False)}
    acc.record_tick(_frame(friendly, enemy, ball_xy=(0.0, 0.0)))
    acc.record_tick(_frame(friendly, enemy, ball_xy=(0.0, 0.0)))

    stats = acc.finalize()
    assert stats.zone_time_pct["friendly_1"]["mid"] == 1.0
    # Enemy robot at x=+4.0 attacks toward +x (own goal at +4.5): attacking zone.
    assert stats.zone_time_pct["enemy_1"]["attacking"] == 1.0


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
    assert data["zone_time_pct"]["friendly_1"]["mid"] == 1.0
    assert data["shots"] == {"friendly": 0, "enemy": 0}


# ---------------------------------------------------------------------------
# Shots / ball travel / robot motion — the "meaningful boxscore" additions
# ---------------------------------------------------------------------------


def _fast_frame(ball_xy, ball_v, my_team_is_right: bool = True) -> GameFrame:
    ball = Ball(Vector3D(ball_xy[0], ball_xy[1], 0), Vector3D(ball_v[0], ball_v[1], 0), None)
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=ball,
    )


def test_shot_counted_once_per_kick_for_friendly():
    # my_team_is_right=True -> own goal at +4.5, friendly attacks toward -x.
    acc = MatchStatsAccumulator()
    fast = _fast_frame((-3.0, 0.0), (-6.0, 0.0))  # hard ball toward enemy goal in attacking half
    slow = _fast_frame((-3.1, 0.0), (-0.1, 0.0))  # same play, now rolling

    acc.record_tick(fast)
    acc.record_tick(fast)  # still fast + locked -> must not double-count
    acc.record_tick(slow)  # lock released
    acc.record_tick(fast)  # second kick -> second shot

    stats = acc.finalize()
    assert stats.shots == {"friendly": 2, "enemy": 0}


def test_no_shot_for_slow_or_backwards_ball():
    acc = MatchStatsAccumulator()
    acc.record_tick(_fast_frame((-3.0, 0.0), (-2.0, 0.0)))  # too slow
    acc.record_tick(_fast_frame((-3.0, 0.0), (6.0, 0.0)))  # moving away from goal
    acc.record_tick(_fast_frame((2.0, 0.0), (-6.0, 0.0)))  # in own half

    stats = acc.finalize()
    assert stats.shots == {"friendly": 0, "enemy": 0}


def test_shot_for_enemy_mirrored():
    acc = MatchStatsAccumulator()
    acc.record_tick(_fast_frame((3.0, 0.0), (6.0, 0.0)))  # enemy attacks +x

    stats = acc.finalize()
    assert stats.shots == {"friendly": 0, "enemy": 1}


def test_no_shot_for_hard_lateral_ball_that_would_miss_the_goal():
    # A cross-field switch or hard clear can satisfy "fast, past midfield,
    # moving in the attacking x-direction" while its actual trajectory is
    # aimed metres wide of the goal mouth (half_goal_width=0.5) -- this must
    # not be counted as a shot. vx=-0.68 dominated by vy=-3.78 means the
    # ball leaves the +/-0.5m goal-mouth band long before reaching the goal
    # line, matching a real case found in `counter_flow_vs_zone_fluid.pkl`.
    acc = MatchStatsAccumulator()
    acc.record_tick(_fast_frame((-2.44, -0.10), (-0.68, -3.78)))

    stats = acc.finalize()
    assert stats.shots == {"friendly": 0, "enemy": 0}


def test_shot_counted_when_trajectory_is_on_target_off_center():
    # Off-center but still within the goal mouth at the goal line should
    # still count -- the fix is about lateral misses, not about requiring
    # dead center.
    acc = MatchStatsAccumulator()
    # From (-3.0, 0.2) with v=(-6.0, 0.3): predicted y at x=-4.5 is
    # 0.2 + 0.3 * ((-4.5 - -3.0) / -6.0) = 0.2 + 0.3*0.25 = 0.275, within 0.5.
    acc.record_tick(_fast_frame((-3.0, 0.2), (-6.0, 0.3)))

    stats = acc.finalize()
    assert stats.shots == {"friendly": 1, "enemy": 0}


def test_no_shot_for_fast_center_field_clearance_even_if_extrapolation_lines_up():
    # Found live in a 2026-09-02 tournament re-run (score_aware_zone_flow_vs_tiki_taka.pkl,
    # t=16.27s): a clearance from essentially the center circle (x=-0.08) at
    # 3.83 m/s toward the enemy goal, extrapolated in a straight line, landed
    # inside the 1m-wide goal mouth 4.5m away purely by chance -- the ball's
    # real velocity collapsed within two ticks (a robot intercepted it), well
    # before it could have reached the goal line. Being in the attacking half
    # is not enough for a long-range straight-line projection to be a
    # meaningful on-target signal; the ball must already be in the attacking
    # third. Matches this file's other tests' shape but at x=-0.08 (attacking
    # half, NOT attacking third: half_length=4.5, _SHOT_ATTACKING_THIRD_M=1.5,
    # so the gate is `progress_from_own_goal > 6.0`, and this ball only
    # reaches ~4.58).
    acc = MatchStatsAccumulator()
    acc.record_tick(_fast_frame((-0.08, -0.09), (-3.83, -0.12)))

    stats = acc.finalize()
    assert stats.shots == {"friendly": 0, "enemy": 0}


def test_shot_still_counted_from_within_the_attacking_third():
    # Same shape as the center-field case above, but from inside the
    # attacking third (x=-2.21, well past the half_length + 1.5m gate) --
    # must still count. Real case from the same replay at t=25.72s (the
    # ball curved off after this tick, but the on-target check only looks
    # at the instantaneous straight-line extrapolation, so this is still a
    # correctly-counted shot attempt).
    acc = MatchStatsAccumulator()
    acc.record_tick(_fast_frame((-2.21, -0.29), (-3.91, -0.01)))

    stats = acc.finalize()
    assert stats.shots == {"friendly": 1, "enemy": 0}


def test_ball_travel_skips_teleport_jumps():
    acc = MatchStatsAccumulator()
    acc.record_tick(_fast_frame((0.0, 0.0), (0.0, 0.0)))
    acc.record_tick(_fast_frame((0.5, 0.0), (0.0, 0.0)))  # 0.5 m of real travel
    acc.record_tick(_fast_frame((3.0, 0.0), (0.0, 0.0)))  # 2.5 m jump = placement, ignored
    acc.record_tick(_fast_frame((3.1, 0.0), (0.0, 0.0)))  # 0.1 m of real travel

    stats = acc.finalize()
    assert stats.ball_travel_m == 0.6


def test_robot_motion_share_requires_velocity_readings():
    acc = MatchStatsAccumulator()
    moving = Robot(id=1, is_friendly=True, has_ball=False, p=Vector2D(0, 0), v=Vector2D(0.5, 0), a=None, orientation=0)
    still = Robot(id=2, is_friendly=True, has_ball=False, p=Vector2D(0, 0), v=Vector2D(0.0, 0), a=None, orientation=0)
    enemy = {5: _robot(5, 4.0, 0.0, False)}
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: moving, 2: still},
        enemy_robots=enemy,
        ball=Ball(Vector3D(0, 0, 0), Vector3D(0, 0, 0), None),
    )
    acc.record_tick(frame)
    acc.record_tick(frame)

    stats = acc.finalize()
    assert stats.robot_motion_pct["friendly_1"] == 1.0
    assert stats.robot_motion_pct["friendly_2"] == 0.0
    assert "enemy_5" not in stats.robot_motion_pct  # no velocity readings (v=None) -> excluded
