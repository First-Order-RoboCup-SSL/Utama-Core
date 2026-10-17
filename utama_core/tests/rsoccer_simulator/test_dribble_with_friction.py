"""A dribbled ball stays at the kicker with and without two-phase ball friction.

rsim holds a dribbled ball with a fixed ODE joint (`holdBall` in vendor/rSim sslrobot.cpp), while
the two-phase friction (`applyBallFriction` in sslworld.cpp) rewrites the ball's velocity and spin
every substep. These tests check that the friction does not break the hold, and that a released
ball still slows to rest on its own.

The friction-on case needs a robosim binary built from vendor/rSim (one with `set_physics_params`);
with the stock binary it skips.
"""

import math

import numpy as np
import pytest

from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise

TICK = 1 / 60
FRONT = 0.1  # the ball's centre, this far in front of the robot's centre, is in the kicker
HOLD_TOLERANCE = 0.15  # m, ball to the robot's front while dribbling
GRAB_TICKS = 20
DRIBBLE_TICKS = 140  # about 2.3 s of driving: forward, then turning while moving
REST_TICKS = 900  # 15 s of game time for the released ball to come to rest

FRICTION_OFF = SimNoise()
FRICTION_ON = SimNoise(ball_roll_decel=(0.3, 0.3), ball_slide_decel=(3.0, 3.0))


def _env(sim_noise) -> SSLStandardEnv:
    try:
        env = SSLStandardEnv(n_robots_blue=1, n_robots_yellow=1, render_mode=None, sim_noise=sim_noise)
    except RuntimeError as e:
        if "set_physics_params" in str(e):
            pytest.skip("needs a robosim binary built from vendor/rSim")
        raise
    env.reset()
    env.teleport_robot(False, 0, -1.0, 0.0, 0.0)
    env.teleport_ball(-1.0 + FRONT, 0.0)
    return env


def _step(env, blue=(0, 0, 0, 0, 0, 0)) -> None:
    env.step({"team_blue": (np.array(blue, dtype=float),), "team_yellow": (np.zeros(6),)})


def _front_gap(env) -> float:
    """Distance from the ball to the point FRONT metres ahead of the robot's centre.

    Both are read from the frame, which rsim keeps in its own coordinates, so the heading and the
    positions are in the same frame.
    """
    r = env.frame.robots_blue[0]
    th = math.radians(r.theta)
    fx, fy = r.x + FRONT * math.cos(th), r.y + FRONT * math.sin(th)
    return math.hypot(env.frame.ball.x - fx, env.frame.ball.y - fy)


def _ball_speed(env) -> float:
    return math.hypot(env.frame.ball.v_x, env.frame.ball.v_y)


@pytest.mark.parametrize("sim_noise", [FRICTION_OFF, FRICTION_ON], ids=["friction_off", "friction_on"])
def test_a_dribbled_ball_stays_at_the_kicker_and_comes_to_rest_after_release(sim_noise):
    env = _env(sim_noise)
    try:
        # Grab: dribbler on, robot still.
        for _ in range(GRAB_TICKS):
            _step(env, (0, 0, 0, 0, 1, 0))
        assert _front_gap(env) < HOLD_TOLERANCE, "the ball should start at the kicker"

        # Dribble: forward for about a second, then turn while still moving forward.
        gaps = []
        for t in range(DRIBBLE_TICKS):
            if t < 60:
                cmd = (1.0, 0.0, 0.0, 0, 1, 0)
            else:
                cmd = (0.5, 0.0, 1.5, 0, 1, 0)
            _step(env, cmd)
            gaps.append(_front_gap(env))
        worst = max(gaps)
        assert worst < HOLD_TOLERANCE, f"the ball got {worst:.3f} m from the kicker while dribbling"

        # Release: dribbler off and robot stopped; the ball must roll to rest by itself.
        speeds = []
        for _ in range(REST_TICKS):
            _step(env, (0, 0, 0, 0, 0, 0))
            speeds.append(_ball_speed(env))
            if len(speeds) > 60 and speeds[-1] < 0.02:
                break
        assert speeds[-1] < 0.02, f"released ball still moving at {speeds[-1]:.3f} m/s after {len(speeds)} ticks"
    finally:
        env.close()
