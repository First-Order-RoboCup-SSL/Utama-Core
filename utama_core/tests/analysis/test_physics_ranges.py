import numpy as np
import pytest

from utama_core.analysis import physics_ranges as pr

DT = pr.TICK_S


def _rolling_ball(n: int, v0: float, decel: float, start: tuple[float, float] = (-3.0, 0.0)):
    """A ball rolling along +x from `start` at `v0`, slowing at a constant `decel` until it stops."""
    t = np.arange(n) * DT
    speed = np.maximum(v0 - decel * t, 0.0)
    x = start[0] + np.cumsum(speed) * DT
    p = np.zeros((n, 3))
    p[:, 0], p[:, 1] = x, start[1]
    v = np.zeros((n, 3))
    v[:, 0] = speed
    return t, p, v


def test_runs_finds_true_ranges_including_the_ends():
    assert pr.runs(np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)) == [(0, 2), (3, 4), (6, 7)]
    assert pr.runs(np.zeros(3, dtype=bool)) == []


def test_window_all_needs_every_tick_of_the_window():
    mask = np.ones(10, dtype=bool)
    mask[5] = False
    out = pr.window_all(mask, 2)
    # t is ok only if [t-2, t+2] avoids tick 5 and the ends
    assert out.tolist() == [False, False, True, False, False, False, False, False, False, False]


def test_kick_ticks_marks_the_first_tick_of_a_speed_jump_only():
    v = np.zeros((60, 3))
    v[20:, 0] = 4.0  # a kick at tick 20
    kicks = pr.kick_ticks(v)
    assert kicks.tolist() == [20]
    # a slow push below KICK_SPEED_MPS is not a kick
    v[:] = 0.0
    v[20:, 0] = pr.KICK_SPEED_MPS - 0.1
    assert pr.kick_ticks(v).tolist() == []


def test_since_last_counts_seconds_from_the_latest_event():
    out = pr.since_last(np.array([2, 5]), 7)
    assert np.isinf(out[:2]).all()
    assert out[2:].tolist() == pytest.approx([0, DT, 2 * DT, 0, DT])


def test_fit_segment_recovers_a_constant_deceleration():
    t = np.arange(60) * DT
    decel, r2 = pr.fit_segment(t, 2.0 - 0.3 * t)
    assert decel == pytest.approx(0.3)
    assert r2 == pytest.approx(1.0)


def test_free_ball_excludes_near_robots_walls_and_chips():
    ball = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.4, 0.0, 0.0], [0.0, 0.0, 0.2]])
    robots = np.full((4, 1, 2), np.nan)
    robots[1, 0] = (pr.FREE_ROBOT_M - 0.01, 0.0)  # too close
    out = pr.free_ball(ball, robots, (9.0, 6.0))
    # free; robot within FREE_ROBOT_M; within WALL_MARGIN_M of the line; in the air
    assert out.tolist() == [True, False, False, False]


def test_ball_seen_flags_the_converters_held_ball_only():
    p = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    v = np.array([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # tick 2: zero velocity and unchanged position (held); tick 3 moved, so it was seen
    assert pr.ball_seen(p, v).tolist() == [True, True, False, True]


def test_ball_deceleration_recovers_rolling_decel_from_velocity_and_positions():
    n = 300
    t, p, v = _rolling_ball(n, 3.0, 0.4)
    free = np.ones(n, dtype=bool)
    live = np.ones(n, dtype=bool)
    out = pr.ball_deceleration(t, p, v, free, live)
    for key in ("by_speed_vel", "by_speed_pos"):
        medians = [row["median"] for row in out[key] if row["n"] >= 10]
        assert medians and all(m == pytest.approx(0.4, abs=0.02) for m in medians)
    (seg,) = out["segments"]  # no kick: one segment, until the ball drops below MIN_SPEED_MPS
    assert seg["decel"] == pytest.approx(0.4, abs=1e-3)
    assert seg["end_mps"] > pr.MIN_SPEED_MPS


def test_rolling_segment_starts_slide_s_after_a_kick():
    n = 240
    t, p, v = _rolling_ball(n, 4.0, 0.3)
    v[:30] = 0.0  # at rest, then kicked to 4 m/s at tick 30
    v[30:, 0] = np.maximum(4.0 - 0.3 * (t[30:] - t[30]), 0)
    free = np.ones(n, dtype=bool)
    out = pr.ball_deceleration(t, p, v, free, np.ones(n, dtype=bool))
    (seg,) = out["segments"]
    expected_start = 4.0 - 0.3 * round(pr.SLIDE_S / DT) * DT
    assert seg["start_mps"] == pytest.approx(expected_start, abs=0.01)


def test_robot_motion_percentiles_for_a_known_ramp():
    n = int(30 / DT)
    t = np.arange(n) * DT
    # accelerate at 2 m/s² to 1.5 m/s, brake at 3 m/s² to 0, repeat
    cycle_v = np.r_[np.arange(0, 1.5, 2 * DT), np.full(60, 1.5), np.arange(1.5, 0, -3 * DT), np.zeros(30)]
    speed = np.resize(cycle_v, n)
    v = np.zeros((n, 1, 2))
    v[:, 0, 0] = speed
    p = np.zeros((n, 1, 2))
    p[:, 0, 0] = np.cumsum(speed) * DT
    orientation = np.zeros((n, 1))
    orientation[:, 0] = (0.5 * t) % (2 * np.pi) - np.pi  # turning at 0.5 rad/s, wrapped
    (m,) = pr.robot_motion(p, v, orientation, np.ones(n, dtype=bool))
    assert m["speed_p99"] == pytest.approx(1.5, abs=0.01)
    assert m["accel_p95"] == pytest.approx(2.0, abs=0.05)
    assert m["brake_p95"] == pytest.approx(3.0, abs=0.05)
    assert m["turn_p95"] == pytest.approx(0.5, abs=0.01)  # the wrap is unwrapped


def test_robot_motion_skips_robots_with_too_little_live_play():
    n = int((pr.MIN_ROBOT_LIVE_S - 1) / DT)
    zeros = np.zeros((n, 1, 2))
    assert pr.robot_motion(zeros, zeros, np.zeros((n, 1)), np.ones(n, dtype=bool)) == [None]


def test_dropouts_counts_short_robot_gaps_but_not_outages_or_substitutions():
    n = 600
    p = np.zeros((n, 2, 2))
    p[100:110, 0] = np.nan  # a 10-tick gap: counted
    p[200:210] = np.nan  # every robot missing: a tracker outage
    p[300:, 1] = np.nan  # gone to the end: a substitution
    seen = np.ones(n, dtype=bool)
    seen[50:53] = False
    out = pr.dropouts(seen, p, np.ones(n, dtype=bool))
    assert out["robot"]["run_s"] == pytest.approx([10 * DT])
    assert out["tracker_outage_s"] == pytest.approx(10 * DT, abs=0.01)
    assert out["ball"]["run_s"] == pytest.approx([3 * DT])
    assert "ball" not in pr.dropouts(None, p, np.ones(n, dtype=bool))
