"""Tests for `utama_core.engine.match_stats` — the aggregate per-match boxscore."""

from __future__ import annotations

import json

import pytest

from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.engine.match_stats import MatchStatsAccumulator
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage


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


# ---------------------------------------------------------------------------
# turnovers / completed_passes / attacking_third_entries -- live equivalents
# of tools/metric_correlation.py's offline definitions (see match_stats.py's
# module docstring and `_update_possession_events`).
# ---------------------------------------------------------------------------


def _poss_frame(ball_xy, ball_v, friendly_robots, enemy_robots, my_team_is_right: bool = True) -> GameFrame:
    ball = Ball(Vector3D(ball_xy[0], ball_xy[1], 0), Vector3D(ball_v[0], ball_v[1], 0), None)
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly_robots,
        enemy_robots=enemy_robots,
        ball=ball,
    )


def test_zero_case_no_possession_or_entry_events():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}
    for _ in range(5):
        acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy))

    stats = acc.finalize()
    assert stats.turnovers == 0
    assert stats.completed_passes == 0
    assert stats.attacking_third_entries == 0


def test_completed_pass_friendly_to_friendly():
    # Robot 1 controls the ball at rest, then a different friendly robot (3)
    # is found in control -- a same-side handoff without an intervening
    # opposing possession is a completed pass.
    acc = MatchStatsAccumulator()
    friendly1 = {1: _robot(1, 0.0, 0.0, True), 3: _robot(3, 5.0, 5.0, True)}
    friendly2 = {1: _robot(1, 5.0, 5.0, True), 3: _robot(3, 1.0, 0.0, True)}
    enemy = {2: _robot(2, -5.0, -5.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly1, enemy))  # robot 1 controls
    acc.record_tick(_poss_frame((1.0, 0.0), (0.0, 0.0), friendly2, enemy))  # robot 3 controls

    stats = acc.finalize()
    assert stats.completed_passes == 1
    assert stats.turnovers == 0


def test_turnover_friendly_to_enemy():
    # Friendly robot 1 controls, then the ball is found controlled by an
    # enemy robot -- possession changed sides -> a turnover.
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy1 = {2: _robot(2, 5.0, 5.0, False)}
    enemy2 = {2: _robot(2, 1.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy1))  # friendly controls
    acc.record_tick(_poss_frame((1.0, 0.0), (0.0, 0.0), friendly, enemy2))  # enemy controls

    stats = acc.finalize()
    assert stats.turnovers == 1
    assert stats.completed_passes == 0


def test_enemy_to_enemy_handoff_counted_as_enemy_completed_pass_not_friendly():
    # Possession moving between two enemy robots must not be tallied as a
    # friendly completed_pass/turnover (MatchStats.turnovers/completed_passes
    # are friendly-only) -- but must be tallied as the enemy counterpart.
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, -5.0, -5.0, True)}
    enemy1 = {2: _robot(2, 0.0, 0.0, False), 4: _robot(4, 5.0, 5.0, False)}
    enemy2 = {2: _robot(2, 5.0, 5.0, False), 4: _robot(4, 1.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy1))  # enemy robot 2 controls
    acc.record_tick(_poss_frame((1.0, 0.0), (0.0, 0.0), friendly, enemy2))  # enemy robot 4 controls

    stats = acc.finalize()
    assert stats.turnovers == 0
    assert stats.completed_passes == 0
    assert stats.enemy_completed_passes == 1
    assert stats.enemy_turnovers == 0


# ---------------------------------------------------------------------------
# pass_distances_m / pass_progress_m -- pass-quality distribution added
# 2026-09-04 (see match_stats.py's module docstring / MatchStats.pass_distances_m
# for the GiveAndGoTactic near-pointless-pass bug this is a detection signal
# for).
# ---------------------------------------------------------------------------


def test_completed_pass_records_distance_and_forward_progress():
    # my_team_is_right=True -> friendly's own goal at +4.5, friendly attacks
    # toward -x. Robot 1 holds at (0.0, 0.0), passes to robot 3 who receives
    # at (-3.0, 4.0): straight-line distance = 5.0m (3-4-5 triangle);
    # progress = (receiver.x - passer.x) * attack_sign = (-3.0 - 0.0) * -1.0
    # = 3.0m toward the opponent goal.
    acc = MatchStatsAccumulator()
    friendly1 = {1: _robot(1, 0.0, 0.0, True), 3: _robot(3, 10.0, 10.0, True)}
    friendly2 = {1: _robot(1, 10.0, 10.0, True), 3: _robot(3, -3.0, 4.0, True)}
    enemy = {2: _robot(2, -5.0, -5.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly1, enemy))  # robot 1 controls
    acc.record_tick(_poss_frame((-2.9, 4.0), (0.0, 0.0), friendly2, enemy))  # robot 3 controls

    stats = acc.finalize()
    assert stats.completed_passes == 1
    assert len(stats.pass_distances_m) == 1
    assert len(stats.pass_progress_m) == 1
    assert stats.pass_distances_m[0] == pytest.approx(5.0, abs=1e-6)
    assert stats.pass_progress_m[0] == pytest.approx(3.0, abs=1e-6)
    # Enemy lists must stay empty -- this was a friendly-only handoff.
    assert stats.enemy_pass_distances_m == []
    assert stats.enemy_pass_progress_m == []


def test_backward_pass_records_negative_progress():
    # Same setup, but the receiver ends up further from the opponent goal
    # than the passer (x increases while friendly attacks -x) -> progress
    # must be negative even though the pass still completes.
    acc = MatchStatsAccumulator()
    friendly1 = {1: _robot(1, -3.0, 0.0, True), 3: _robot(3, 10.0, 10.0, True)}
    friendly2 = {1: _robot(1, 10.0, 10.0, True), 3: _robot(3, 0.0, 0.0, True)}
    enemy = {2: _robot(2, -5.0, -5.0, False)}

    acc.record_tick(_poss_frame((-3.0, 0.0), (0.0, 0.0), friendly1, enemy))  # robot 1 controls
    acc.record_tick(_poss_frame((0.1, 0.0), (0.0, 0.0), friendly2, enemy))  # robot 3 controls

    stats = acc.finalize()
    assert stats.completed_passes == 1
    assert stats.pass_distances_m[0] == pytest.approx(3.0, abs=1e-6)
    assert stats.pass_progress_m[0] == pytest.approx(-3.0, abs=1e-6)


def test_enemy_completed_pass_populates_enemy_lists_not_friendly():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, -5.0, -5.0, True)}
    enemy1 = {2: _robot(2, 0.0, 0.0, False), 4: _robot(4, 10.0, 10.0, False)}
    enemy2 = {2: _robot(2, 10.0, 10.0, False), 4: _robot(4, 2.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy1))  # enemy robot 2 controls
    acc.record_tick(_poss_frame((2.0, 0.0), (0.0, 0.0), friendly, enemy2))  # enemy robot 4 controls

    stats = acc.finalize()
    assert stats.enemy_completed_passes == 1
    assert len(stats.enemy_pass_distances_m) == 1
    assert len(stats.enemy_pass_progress_m) == 1
    assert stats.enemy_pass_distances_m[0] == pytest.approx(2.0, abs=1e-6)
    # my_team_is_right=True -> enemy's own goal at -4.5, enemy attacks +x ->
    # attack_sign = own_goal_sign = 1.0 -> progress = (2.0 - 0.0) * 1.0 = 2.0.
    assert stats.enemy_pass_progress_m[0] == pytest.approx(2.0, abs=1e-6)
    assert stats.pass_distances_m == []
    assert stats.pass_progress_m == []


def test_turnover_does_not_add_a_pass_quality_entry():
    # Mirror of test_turnover_friendly_to_enemy -- a turnover (possession
    # crossing sides) must not be recorded in either pass list.
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy1 = {2: _robot(2, 5.0, 5.0, False)}
    enemy2 = {2: _robot(2, 1.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy1))  # friendly controls
    acc.record_tick(_poss_frame((1.0, 0.0), (0.0, 0.0), friendly, enemy2))  # enemy controls

    stats = acc.finalize()
    assert stats.turnovers == 1
    assert stats.pass_distances_m == []
    assert stats.pass_progress_m == []
    assert stats.enemy_pass_distances_m == []
    assert stats.enemy_pass_progress_m == []


def test_pass_quality_lists_serialize_to_json(tmp_path):
    acc = MatchStatsAccumulator()
    friendly1 = {1: _robot(1, 0.0, 0.0, True), 3: _robot(3, 10.0, 10.0, True)}
    friendly2 = {1: _robot(1, 10.0, 10.0, True), 3: _robot(3, -3.0, 4.0, True)}
    enemy = {2: _robot(2, -5.0, -5.0, False)}
    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly1, enemy))
    acc.record_tick(_poss_frame((-2.9, 4.0), (0.0, 0.0), friendly2, enemy))

    out = tmp_path / "stats.json"
    acc.finalize().to_json(out)

    data = json.loads(out.read_text())
    assert len(data["pass_distances_m"]) == 1
    assert data["pass_distances_m"][0] == pytest.approx(5.0, abs=1e-6)
    assert len(data["pass_progress_m"]) == 1
    assert data["enemy_pass_distances_m"] == []
    assert data["enemy_pass_progress_m"] == []


def test_pass_resolved_after_release_at_speed():
    # Robot 1 controls, releases the ball at speed (a real pass), and it's
    # picked up again by a different friendly robot once it slows -- must
    # still resolve as one completed pass via the release->reacquire path.
    acc = MatchStatsAccumulator()
    friendly_near1 = {1: _robot(1, 0.0, 0.0, True), 3: _robot(3, 5.0, 5.0, True)}
    friendly_mid = {1: _robot(1, -5.0, 0.0, True), 3: _robot(3, 5.0, 5.0, True)}
    friendly_near3 = {1: _robot(1, -5.0, 0.0, True), 3: _robot(3, 1.0, 0.0, True)}
    enemy = {2: _robot(2, -5.0, -5.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly_near1, enemy))  # robot 1 controls
    acc.record_tick(_poss_frame((0.5, 0.0), (6.0, 0.0), friendly_mid, enemy))  # released at speed
    acc.record_tick(_poss_frame((0.9, 0.0), (6.0, 0.0), friendly_mid, enemy))  # in flight, no one near
    acc.record_tick(_poss_frame((1.0, 0.0), (0.0, 0.0), friendly_near3, enemy))  # robot 3 controls, slow

    stats = acc.finalize()
    assert stats.completed_passes == 1
    assert stats.turnovers == 0


def test_attacking_third_entry_counted_once_with_hysteresis():
    # my_team_is_right=True -> friendly's own goal at +4.5, friendly attacks
    # toward -x; entry threshold is x < -1.5 (half_length + 1.5m third).
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy))  # midfield, not yet entered
    acc.record_tick(_poss_frame((-2.0, 0.0), (0.0, 0.0), friendly, enemy))  # crosses into attacking third
    acc.record_tick(_poss_frame((-1.6, 0.0), (0.0, 0.0), friendly, enemy))  # still inside hysteresis band
    acc.record_tick(_poss_frame((-2.5, 0.0), (0.0, 0.0), friendly, enemy))  # still in third, no re-entry

    stats = acc.finalize()
    assert stats.attacking_third_entries == 1


def test_attacking_third_re_entry_after_full_exit():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy))  # midfield
    acc.record_tick(_poss_frame((-2.0, 0.0), (0.0, 0.0), friendly, enemy))  # 1st entry
    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy))  # exits past hysteresis band
    acc.record_tick(_poss_frame((-2.0, 0.0), (0.0, 0.0), friendly, enemy))  # 2nd entry

    stats = acc.finalize()
    assert stats.attacking_third_entries == 2


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


# ---------------------------------------------------------------------------
# Opponent counterparts / possession-under-pressure / restart-to-entry --
# live ports of tools/metric_correlation.py's metrics 3/4/5, see roadmap
# item 14 ("Metric design: derive, don't invent").
# ---------------------------------------------------------------------------


def _ts_frame(
    ts: float,
    ball_xy,
    ball_v,
    friendly_robots,
    enemy_robots,
    referee=None,
    my_team_is_right: bool = True,
) -> GameFrame:
    ball = Ball(Vector3D(ball_xy[0], ball_xy[1], 0), Vector3D(ball_v[0], ball_v[1], 0), None)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly_robots,
        enemy_robots=enemy_robots,
        ball=ball,
        referee=referee,
    )


def _referee(command: RefereeCommand, ts: float) -> RefereeData:
    return RefereeData(
        source_identifier=None,
        time_sent=ts,
        time_received=ts,
        referee_command=command,
        referee_command_timestamp=ts,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue"),
        yellow_team=TeamInfo(name="yellow"),
    )


def test_friendly_turnover_to_enemy_counts_enemy_side_zero():
    """Sanity check the friendly-turnover test above still leaves the enemy
    counterpart at zero -- a turnover only ever increments one side."""
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy1 = {2: _robot(2, 5.0, 5.0, False)}
    enemy2 = {2: _robot(2, 1.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy1))
    acc.record_tick(_poss_frame((1.0, 0.0), (0.0, 0.0), friendly, enemy2))

    stats = acc.finalize()
    assert stats.turnovers == 1
    assert stats.enemy_turnovers == 0
    assert stats.enemy_completed_passes == 0


def test_enemy_turnover_to_friendly_counted_on_enemy_side():
    # Mirror of test_turnover_friendly_to_enemy: enemy robot 2 controls, then
    # a friendly robot is found in control -- a turnover attributed to enemy.
    acc = MatchStatsAccumulator()
    friendly1 = {1: _robot(1, 5.0, 5.0, True)}
    friendly2 = {1: _robot(1, 1.0, 0.0, True)}
    enemy = {2: _robot(2, 0.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly1, enemy))  # enemy controls
    acc.record_tick(_poss_frame((1.0, 0.0), (0.0, 0.0), friendly2, enemy))  # friendly controls

    stats = acc.finalize()
    assert stats.enemy_turnovers == 1
    assert stats.enemy_completed_passes == 0
    assert stats.turnovers == 0


def test_enemy_attacking_third_entry_counted_independently_of_friendly():
    # my_team_is_right=True -> enemy's own goal at -4.5, enemy attacks toward
    # +x; enemy's entry threshold is x > 1.5 (half_length + 1.5m third).
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, -4.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}

    acc.record_tick(_poss_frame((0.0, 0.0), (0.0, 0.0), friendly, enemy))  # midfield
    acc.record_tick(_poss_frame((2.0, 0.0), (0.0, 0.0), friendly, enemy))  # enemy enters its attacking third

    stats = acc.finalize()
    assert stats.enemy_attacking_third_entries == 1
    assert stats.attacking_third_entries == 0  # friendly never entered its own


def test_possession_under_pressure_accumulates_seconds_while_contested():
    # Friendly controls the ball (within possession radius, ball slow) while
    # an enemy sits within the pressure radius -- seconds accumulate as real
    # elapsed sim time (ts deltas), not a tick count.
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy_close = {2: _robot(2, 0.3, 0.0, False)}  # within _PRESSURE_RADIUS_M (0.5m) of the ball
    enemy_far = {2: _robot(2, 5.0, 0.0, False)}

    acc.record_tick(_ts_frame(0.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy_close))
    acc.record_tick(_ts_frame(0.5, (0.0, 0.0), (0.0, 0.0), friendly, enemy_close))
    acc.record_tick(_ts_frame(1.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy_far))  # pressure released

    stats = acc.finalize()
    # Only the 0.0->0.5s tick-to-tick gap was under pressure (the first tick
    # has no prior ts to diff against, so contributes 0s).
    assert stats.possession_under_pressure_s["friendly"] == pytest.approx(0.5, abs=1e-6)
    assert stats.possession_under_pressure_s["enemy"] == 0.0


def test_possession_under_pressure_zero_when_opponent_out_of_range():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy_far = {2: _robot(2, 5.0, 0.0, False)}

    acc.record_tick(_ts_frame(0.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy_far))
    acc.record_tick(_ts_frame(0.5, (0.0, 0.0), (0.0, 0.0), friendly, enemy_far))

    stats = acc.finalize()
    assert stats.possession_under_pressure_s == {"friendly": 0.0, "enemy": 0.0}


def test_restart_to_first_entry_resolves_on_possessing_sides_own_entry():
    # A live-play command starts right after a non-live one (kickoff ending)
    # with friendly nearest the ball -- the clock is attributed to friendly
    # and stops once friendly's own attacking-third-entry hysteresis fires.
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}

    acc.record_tick(_ts_frame(0.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy, referee=_referee(RefereeCommand.STOP, 0.0)))
    acc.record_tick(
        _ts_frame(1.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy, referee=_referee(RefereeCommand.NORMAL_START, 1.0))
    )  # restart clock starts, friendly nearest the ball
    acc.record_tick(
        _ts_frame(3.5, (-2.0, 0.0), (0.0, 0.0), friendly, enemy, referee=_referee(RefereeCommand.NORMAL_START, 3.5))
    )  # friendly's attacking third entry (x < -1.5) -- resolves the clock

    stats = acc.finalize()
    assert stats.n_restarts == 1
    assert stats.n_restarts_with_entry == 1
    assert stats.restart_to_first_entry_s["friendly"] == pytest.approx(2.5, abs=1e-6)
    assert stats.restart_to_first_entry_s["enemy"] is None


def test_restart_with_no_entry_excluded_from_mean():
    acc = MatchStatsAccumulator()
    friendly = {1: _robot(1, 0.0, 0.0, True)}
    enemy = {2: _robot(2, 4.0, 0.0, False)}

    acc.record_tick(_ts_frame(0.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy, referee=_referee(RefereeCommand.STOP, 0.0)))
    acc.record_tick(
        _ts_frame(1.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy, referee=_referee(RefereeCommand.NORMAL_START, 1.0))
    )
    # Never reaches friendly's attacking third before the match ends.
    acc.record_tick(
        _ts_frame(5.0, (0.0, 0.0), (0.0, 0.0), friendly, enemy, referee=_referee(RefereeCommand.NORMAL_START, 5.0))
    )

    stats = acc.finalize()
    assert stats.n_restarts == 1
    assert stats.n_restarts_with_entry == 0
    assert stats.restart_to_first_entry_s["friendly"] is None
    assert "enemy_5" not in stats.robot_motion_pct  # no velocity readings (v=None) -> excluded


def test_fouls_by_side_uses_offending_teams_or_the_restart_colour():
    """We are yellow. `offending_teams` wins where a rule sets it; rules that
    don't (out of bounds, double touch, ...) are charged to the side that did
    *not* get the restart; a goal is nobody's foul."""
    acc = MatchStatsAccumulator()

    def v(rule, next_command=None, offending_teams=()):
        return RuleViolation(
            rule_name=rule,
            suggested_command=RefereeCommand.STOP,
            next_command=next_command,
            status_message="",
            offending_teams=offending_teams,
        )

    acc.record_rule_violation(v("excessive_dribbling", RefereeCommand.DIRECT_FREE_BLUE, (True,)), True)
    acc.record_rule_violation(v("double_touch", RefereeCommand.DIRECT_FREE_BLUE), True)  # blue's restart -> ours
    acc.record_rule_violation(v("out_of_bounds", RefereeCommand.BALL_PLACEMENT_YELLOW), True)  # theirs
    acc.record_rule_violation(v("crashing", RefereeCommand.FORCE_START, (True, False)), True)  # both
    acc.record_rule_violation(v("goal", RefereeCommand.PREPARE_KICKOFF_BLUE), True)  # nobody's
    acc.record_rule_violation(v("double_touch", RefereeCommand.DIRECT_FREE_BLUE))  # side unknown: tally only

    stats = acc.finalize()
    assert stats.fouls_by_side == {
        "friendly": {"excessive_dribbling": 1, "double_touch": 1, "crashing": 1},
        "enemy": {"out_of_bounds": 1, "crashing": 1},
    }
    assert stats.rule_event_counts["double_touch"] == 2
