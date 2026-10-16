"""`SimNoise`'s physics knobs: two-phase ball friction, kick direction, substeps, motion limits.

The rsim tests of the C++ knobs need a robosim binary built from `vendor/rSim` (one with
`set_physics_params`); with the stock rc-robosim binary they skip.
"""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from utama_core.rsoccer_simulator.src.Entities import Robot
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from utama_core.rsoccer_simulator.src.Utils.sim_noise import (
    KickAngleNoise,
    MotionLimits,
    SimDraw,
    SimNoise,
)

N = {"team_blue": 2, "team_yellow": 2}
TICK = 1 / 60

# --- settings and draws ------------------------------------------------------------------


def test_with_the_new_knobs_off_the_recorded_settings_and_draws_are_what_they_were():
    """Match keys and replays from before the knobs existed must stay the same."""
    noise = SimNoise(seed=4, latency_s=(0.02, 0.05), kick_speed_offset_sd=0.2)
    assert set(noise.to_dict()) == {"seed", "latency_s", "kick_speed_offset_sd", "kick_speed_sd", "vision", "vanishing"}
    assert set(noise.draw(N, TICK).to_dict()) == {"latency_ticks", "kick_offset"}
    assert not noise.native_physics and not noise.motion_limits


def test_turning_a_new_knob_on_leaves_the_older_draws_alone():
    old = SimNoise(seed=4, latency_s=(0.02, 0.05), kick_speed_offset_sd=0.2)
    new = dataclasses.replace(old, ball_roll_decel=(0.28, 0.33), max_speed=(1.0, 2.0), kick_angle_offset_sd_deg=3.0)
    a, b = old.draw(N, TICK), new.draw(N, TICK)
    assert (a.latency_ticks, a.kick_offset) == (b.latency_ticks, b.kick_offset)


def test_new_knobs_round_trip_and_repeat_with_the_seed():
    noise = SimNoise(
        seed=7,
        ball_roll_decel=(0.28, 0.33),
        ball_slide_decel=(2.0, 4.0),
        kick_angle_offset_sd_deg=2.0,
        kick_angle_sd_deg=1.0,
        max_speed=(1.5, 2.0),
        max_accel=(2.0, 4.0),
        max_brake=(2.0, 5.0),
        substeps=4,
    )
    assert SimNoise.from_dict(noise.to_dict()) == noise
    d = noise.draw(N, TICK)
    assert d == noise.draw(N, TICK)
    assert d != dataclasses.replace(noise, seed=8).draw(N, TICK)
    assert 0.28 <= d.ball_roll_decel <= 0.33 and 2.0 <= d.ball_slide_decel <= 4.0
    assert all(1.5 <= v <= 2.0 for v in d.max_speed["team_blue"] + d.max_speed["team_yellow"])
    assert len(set(d.max_accel["team_blue"])) == 2, "drawn per robot"
    assert noise.native_physics and noise.motion_limits


def test_sliding_needs_a_rolling_deceleration():
    with pytest.raises(AssertionError):
        SimNoise(ball_slide_decel=(2.0, 4.0))


# --- kick direction ------------------------------------------------------------------------


def test_a_held_kick_keeps_one_direction_and_the_next_kick_draws_again():
    cfg = SimNoise(seed=1)
    noise = KickAngleNoise({"team_blue": (5.0, 0.0), "team_yellow": (0.0, 0.0)}, 2.0, cfg.kick_angle_rngs(N))
    first = [noise.angle("team_blue", 0, True) for _ in range(3)]
    assert len(set(first)) == 1
    assert noise.angle("team_blue", 0, False) == 0.0
    second = noise.angle("team_blue", 0, True)
    assert second != first[0]
    many = []
    for _ in range(2000):
        many.append(noise.angle("team_blue", 0, True))
        noise.angle("team_blue", 0, False)
    assert np.degrees(np.mean(many)) == pytest.approx(5.0, abs=0.2)
    assert np.degrees(np.std(many)) == pytest.approx(2.0, abs=0.2)


def test_an_offset_alone_kicks_every_time_at_the_offset():
    cfg = SimNoise(seed=1)
    noise = KickAngleNoise({"team_blue": (-3.0, 0.0), "team_yellow": (0.0, 0.0)}, 0.0, cfg.kick_angle_rngs(N))
    assert noise.angle("team_blue", 0, True) == pytest.approx(math.radians(-3.0))


# --- motion limits -------------------------------------------------------------------------


def _limits(speed=None, accel=None, brake=None) -> MotionLimits:
    def per(v):
        return None if v is None else {"team_blue": (v, v), "team_yellow": (v, v)}

    draw = SimDraw(
        latency_ticks={}, kick_offset={}, max_speed=per(speed), max_accel=per(accel), max_brake=per(brake)
    )
    return MotionLimits(draw, 0.1)


def _limited(limits, cmd, vel, theta_deg=0.0):
    command = Robot(v_x=cmd[0], v_y=cmd[1])
    limits.apply("team_blue", 0, command, SimpleNamespace(v_x=vel[0], v_y=vel[1], theta=theta_deg))
    return command.v_x, command.v_y


def test_speeding_up_is_capped_at_accel_times_dt():
    assert _limited(_limits(accel=2.0), (1.0, 0.0), (0.0, 0.0)) == pytest.approx((0.2, 0.0))
    # exactly at the cap: unchanged
    assert _limited(_limits(accel=2.0), (0.2, 0.0), (0.0, 0.0)) == pytest.approx((0.2, 0.0))


def test_slowing_down_is_capped_by_the_braking_rate_not_the_acceleration():
    limits = _limits(accel=1.0, brake=3.0)
    assert _limited(limits, (0.0, 0.0), (1.0, 0.0)) == pytest.approx((0.7, 0.0))


def test_top_speed_is_capped_in_any_direction():
    vx, vy = _limited(_limits(speed=1.0), (2.0, 2.0), (0.0, 0.0))
    assert math.hypot(vx, vy) == pytest.approx(1.0)
    assert vx == pytest.approx(vy)


def test_limits_compare_the_local_command_with_the_world_velocity_by_the_robots_heading():
    """Facing +y and moving +y at 1 m/s, a forward command of 1 m/s is no change at all."""
    assert _limited(_limits(accel=0.1), (1.0, 0.0), (0.0, 1.0), theta_deg=90.0) == pytest.approx((1.0, 0.0))
    # Facing +y but moving +x: the command must turn the velocity, so it is capped.
    vx, vy = _limited(_limits(accel=1.0), (1.0, 0.0), (1.0, 0.0), theta_deg=90.0)
    world = (-vy, vx)  # local -> world for a 90 degree heading
    assert math.hypot(world[0] - 1.0, world[1]) == pytest.approx(0.1)


def test_no_limits_leave_the_command_alone():
    assert _limited(_limits(), (3.0, -1.0), (0.0, 0.0), theta_deg=37.0) == pytest.approx((3.0, -1.0))


# --- in rsim -------------------------------------------------------------------------------


def _env(sim_noise=None) -> SSLStandardEnv:
    try:
        env = SSLStandardEnv(n_robots_blue=1, n_robots_yellow=1, render_mode=None, sim_noise=sim_noise)
    except RuntimeError as e:
        if "set_physics_params" in str(e):
            pytest.skip("needs a robosim binary built from vendor/rSim")
        raise
    env.reset()
    env.teleport_robot(True, 0, -3.0, -2.5)
    return env


def _step(env, blue=(0, 0, 0, 0, 0, 0)) -> None:
    env.step({"team_blue": (np.array(blue, dtype=float),), "team_yellow": (np.zeros(6),)})


def _ball_speed(env) -> float:
    return math.hypot(env.frame.ball.v_x, env.frame.ball.v_y)


def _kick(env, theta=0.0, ticks=30) -> list:
    """Blue 0 at (-2, 0) facing `theta` kicks a ball off its dribbler; ball velocity per tick."""
    env.teleport_robot(False, 0, -2.0, 0.0, theta)
    env.teleport_ball(-2.0 + 0.1 * math.cos(theta), 0.1 * math.sin(theta))
    for _ in range(10):
        _step(env, (0, 0, 0, 0, 1, 0))
    _step(env, (0, 0, 0, 1, 0, 0))
    out = []
    for _ in range(ticks):
        _step(env)
        out.append((env.frame.ball.v_x, env.frame.ball.v_y))
    return out


def test_a_rolling_ball_slows_at_the_set_rolling_deceleration():
    env = _env(SimNoise(ball_roll_decel=(0.3, 0.3)))
    try:
        env.teleport_robot(False, 0, -3.0, 2.5)
        env.teleport_ball(-2.0, 0.0, 2.0, 0.0)
        speeds = []
        for _ in range(60):
            _step(env)
            speeds.append(_ball_speed(env))
        # The first tick reports the ball still; rsim advances int(1000/60) = 16 ms of physics a tick
        assert (speeds[1] - speeds[59]) / (58 * 0.016) == pytest.approx(0.3, rel=0.01)
    finally:
        env.close()


def test_a_kicked_ball_slides_then_rolls_from_five_sevenths_of_its_kick_speed():
    env = _env(SimNoise(ball_roll_decel=(0.3, 0.3), ball_slide_decel=(3.0, 3.0)))
    try:
        speeds = [math.hypot(*v) for v in _kick(env, ticks=60)]
        decel = -np.diff(speeds) / 0.016
        assert np.median(decel[3:15]) == pytest.approx(3.0, rel=0.02)
        assert np.median(decel[-15:]) == pytest.approx(0.3, rel=0.02)
        slide_end = speeds[int(np.argmax(decel[3:] < 1.0)) + 3]
        assert slide_end == pytest.approx(5 / 7 * speeds[0], rel=0.05)
    finally:
        env.close()


def test_kicks_leave_at_the_robots_drawn_direction_offset():
    straight = _env()
    noisy = _env(SimNoise(seed=2, kick_angle_offset_sd_deg=10.0))
    try:
        offset = noisy.sim_draw.kick_angle_offset["team_blue"][0]
        assert abs(offset) > 5, "the seed should give a clear offset"
        a0 = math.degrees(math.atan2(*reversed(_kick(straight)[5])))
        a1 = math.degrees(math.atan2(*reversed(_kick(noisy)[5])))
        assert abs(a0) < 1
        # rsim's frame is the mirror of ours; the angle is the deflection in rsim's frame
        assert a1 - a0 == pytest.approx(offset, abs=2.0)
    finally:
        straight.close()
        noisy.close()


def _drive(env, ticks=40) -> list:
    env.teleport_robot(False, 0, -2.0, 1.0, 0.0)
    out = []
    for _ in range(ticks):
        _step(env, (2.0, 0.5, 1.0, 0, 0, 0))
        r = env.frame.robots_blue[0]
        out.append((r.x, r.y, r.theta))
    return out


def test_substeps_change_the_physics_and_five_is_rsims_own():
    runs = []
    for substeps in (None, 5, 3):
        env = _env(SimNoise(substeps=3) if substeps == 3 else None)
        try:
            if substeps == 5:
                # what the env sends when another knob is on; with substeps 5 it must change nothing
                try:
                    env.rsim.set_physics_params(
                        substeps=5, ball_slide_decel=-1, ball_roll_decel=-1, ball_slide_fraction=5 / 7
                    )
                except RuntimeError:
                    pytest.skip("needs a robosim binary built from vendor/rSim")
            runs.append(_drive(env))
        finally:
            env.close()
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]
    assert math.dist(runs[0][-1][:2], runs[2][-1][:2]) < 0.05, "fewer substeps: same motion, roughly"


def test_motion_limits_cap_speed_and_acceleration_under_any_control():
    env = _env(SimNoise(max_speed=(1.0, 1.0), max_accel=(1.0, 1.0)))
    try:
        env.teleport_robot(False, 0, -2.0, 1.0, 1.2)
        speeds = []
        for _ in range(90):
            _step(env, (0.5, 2.0, 0, 0, 0, 0))
            r = env.frame.robots_blue[0]
            speeds.append(math.hypot(r.v_x, r.v_y))
        # 1 m/s^2, less 4% (rsim steps 16 ms of physics per 1/60 s tick) and a little wheel lag;
        # unlimited, rsim's own motors reach about 3 m/s^2
        assert 0.85 <= (speeds[40] - speeds[10]) / (30 * TICK) <= 1.0
        assert max(speeds) <= 1.01
    finally:
        env.close()
