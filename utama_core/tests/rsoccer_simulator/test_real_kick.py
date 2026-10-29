"""rsim kicks a robot with a calibrated kicker (`RealKick`) at its measured ball speed, off its
heading by its bias, bent towards the side the ball sits on the dribbler, scattered by its spread;
a robot without one keeps the sim's kick."""

import math

import numpy as np
import pytest

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.robot_params import RSIM_PARAMS
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.rsoccer_simulator.src.Utils.real_kick import RealKick, RealKicks
from utama_core.team_controller.src.controllers.common.robot_profile import RobotProfile

STILL = {"team_yellow": (np.zeros(5, np.float32),), "team_blue": ()}
KICK = {"team_yellow": (np.array([0, 0, 0, 1, 0], np.float32),), "team_blue": ()}


@pytest.fixture(scope="module")
def env():
    env = SSLStandardEnv(n_robots_yellow=1, n_robots_blue=0, render_mode=None)
    env.reset()
    return env


def kick(env, real: RealKick = None, ball_left=0.0):
    """(direction in degrees anticlockwise of the robot's heading, top speed) of one kick."""
    env.real_kicks = RealKicks({("team_yellow", 0): real}) if real else None
    env.teleport_robot(True, 0, 0.0, 0.0, 0.0)
    env.teleport_ball(ROBOT_RADIUS + 0.025, ball_left)
    env.step(KICK)
    speeds = []
    for _ in range(10):
        env.step(STILL)
        speeds.append(math.hypot(env.frame.ball.v_x, env.frame.ball.v_y))
    b = env.frame.ball  # the sim's y is ours mirrored
    return math.degrees(math.atan2(-b.v_y, b.v_x)), max(speeds)


def test_the_ball_leaves_at_the_robots_measured_speed(env):
    assert kick(env, RealKick(speed=2.19))[1] == pytest.approx(2.19, abs=0.03)
    assert kick(env)[1] == pytest.approx(RSIM_PARAMS.KICK_SPD, abs=0.3)  # no profile: the sim's


def test_bias_and_bend_turn_the_kick_anticlockwise_towards_the_balls_side(env):
    assert kick(env, RealKick(bias_deg=20.0))[0] == pytest.approx(20.0, abs=1.5)
    assert kick(env, RealKick(bend_deg_per_cm=5.0), ball_left=0.02)[0] == pytest.approx(10.0, abs=1.5)
    assert kick(env, RealKick(bend_deg_per_cm=5.0), ball_left=-0.02)[0] == pytest.approx(-10.0, abs=1.5)


def test_spread_scatters_kicks_by_its_sd_the_same_way_for_a_seed():
    def angles(seed):
        kicks = RealKicks({("team_yellow", 0): RealKick(spread_deg=9.0)}, seed=seed)
        out = []
        for _ in range(400):
            out.append(math.degrees(kicks.angle(("team_yellow", 0), True, 0.0)))
            kicks.angle(("team_yellow", 0), False, 0.0)  # the kick command ends
        return out

    assert np.std(angles(0)) == pytest.approx(9.0, rel=0.15)
    assert angles(0) == angles(0) and angles(0) != angles(1)


def test_a_kick_held_over_several_ticks_keeps_one_direction():
    kicks = RealKicks({("team_yellow", 0): RealKick(spread_deg=9.0)})
    first = kicks.angle(("team_yellow", 0), True, 0.0)
    assert kicks.angle(("team_yellow", 0), True, 0.03) == first


def test_only_a_profile_that_measured_its_kick_has_one():
    assert RealKick.of(RobotProfile()) is None
    assert RealKick.of(RobotProfile(kick_bias_deg=2.7, kick_speed=2.19)) == RealKick(2.19, 0.0, 2.7, 0.0)
