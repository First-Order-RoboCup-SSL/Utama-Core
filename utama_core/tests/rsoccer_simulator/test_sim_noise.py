"""`SimNoise`: seeded latency, kick speed noise and vision noise in rsim."""

import numpy as np
import pytest

from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise
from utama_core.rsoccer_simulator.src.Utils.sim_noise import (
    CommandDelay,
    KickSpeedNoise,
    SimNoise,
)

N = {"team_blue": 2, "team_yellow": 2}
TICK = 1 / 60


def _action(tick: int) -> dict:
    """Commands whose first field is the tick they were issued on, so a delay is readable."""
    return {c: tuple(np.array([tick + 1.0, 0, 0, 0, 0, 0]) for _ in range(2)) for c in ("team_blue", "team_yellow")}


# --- latency -------------------------------------------------------------------------


def test_a_robot_acts_on_its_command_from_exactly_its_latency_ago():
    delay = CommandDelay({"team_blue": (3, 0), "team_yellow": (0, 1)})
    out = [delay.push(_action(t)) for t in range(6)]
    # blue 0, 3 ticks: zeros (stand still) for ticks 0-2, then tick t-3's command
    assert [o["team_blue"][0][0] for o in out] == [0, 0, 0, 1, 2, 3]
    assert [o["team_blue"][1][0] for o in out] == [1, 2, 3, 4, 5, 6]
    assert [o["team_yellow"][1][0] for o in out] == [0, 1, 2, 3, 4, 5]


def test_reset_drops_the_queued_commands():
    delay = CommandDelay({"team_blue": (1, 1), "team_yellow": (1, 1)})
    delay.push(_action(0))
    delay.reset()
    assert delay.push(_action(1))["team_blue"][0][0] == 0


# --- kick speed ----------------------------------------------------------------------


def _kick_noise(offset: float, sd: float, seed: int = 0) -> KickSpeedNoise:
    cfg = SimNoise(seed=seed)
    return KickSpeedNoise({"team_blue": (offset, 0.0), "team_yellow": (0.0, 0.0)}, sd, cfg.kick_rngs(N))


def test_a_held_kick_command_keeps_one_speed_and_the_next_kick_draws_again():
    noise = _kick_noise(0.0, 0.5)
    first = [noise.speed("team_blue", 0, True, 5.0) for _ in range(4)]
    assert len(set(first)) == 1
    assert noise.speed("team_blue", 0, False, 5.0) == 0.0
    assert noise.speed("team_blue", 0, True, 5.0) != first[0]


def test_kick_speeds_centre_on_nominal_plus_the_robots_offset_with_the_given_spread():
    noise = _kick_noise(-0.4, 0.3)
    speeds = []
    for _ in range(4000):
        speeds.append(noise.speed("team_blue", 0, True, 5.0))
        noise.speed("team_blue", 0, False, 5.0)
    assert np.mean(speeds) == pytest.approx(4.6, abs=0.02)
    assert np.std(speeds) == pytest.approx(0.3, abs=0.02)


def test_kick_speed_never_drops_below_the_floor():
    noise = _kick_noise(-10.0, 0.0)
    assert noise.speed("team_blue", 0, True, 5.0) == 0.5


# --- the per-match draw ----------------------------------------------------------------


def test_same_seed_same_draw_and_another_seed_another():
    cfg = SimNoise(seed=7, latency_s=(0.0, 0.2), kick_speed_offset_sd=0.5)
    assert cfg.draw(N, TICK) == SimNoise.from_dict(cfg.to_dict()).draw(N, TICK)
    assert cfg.draw(N, TICK) != SimNoise(**{**cfg.__dict__, "seed": 8}).draw(N, TICK)


def test_latency_is_rounded_to_whole_ticks():
    draw = SimNoise(latency_s=(0.05, 0.05)).draw(N, TICK)
    assert draw.latency_ticks == {"team_blue": (3, 3), "team_yellow": (3, 3)}


def test_defaults_draw_nothing():
    draw = SimNoise().draw(N, TICK)
    assert all(t == 0 for ticks in draw.latency_ticks.values() for t in ticks)
    assert all(o == 0 for offs in draw.kick_offset.values() for o in offs)


# --- in the env (rsim) --------------------------------------------------------------------


def _env(sim_noise=None, **kwargs) -> SSLStandardEnv:
    env = SSLStandardEnv(n_robots_blue=2, n_robots_yellow=2, render_mode=None, sim_noise=sim_noise, **kwargs)
    env.reset()
    return env


def _observed(env) -> list:
    vision = env._frame_to_observations()[0]
    return [(b.x, b.y) for b in vision.balls] + [
        (r.x, r.y, r.orientation) for r in vision.blue_robots + vision.yellow_robots
    ]


def _step_forward(env) -> None:
    cmd = tuple(np.array([1.0, 0, 0, 0, 0, 0]) for _ in range(2))
    env.step({"team_blue": cmd, "team_yellow": cmd})


@pytest.mark.parametrize("noise_kwargs", [{"gaussian_noise": RsimGaussianNoise(0.01, 0.01, 1)}, {}])
def test_vision_noise_leaves_the_sims_own_frame_alone(noise_kwargs):
    """The noise used to be added to `env.frame` in place, and teleports write that frame back
    into the physics, so every ball placement moved every robot by its noise."""
    sim_noise = None if noise_kwargs else SimNoise(seed=1, vision=RsimGaussianNoise(0.01, 0.01, 1))
    env = _env(sim_noise, **noise_kwargs)
    try:
        ball, robot = env.frame.ball, env.frame.robots_blue[0]
        before = (ball.x, ball.y, robot.x, robot.y, robot.theta)
        env.latest_observation = (-1, None)
        observed = _observed(env)
        assert (ball.x, ball.y, robot.x, robot.y, robot.theta) == before
        assert observed[0] != (ball.x, -ball.y)
    finally:
        env.close()


def test_seeded_vision_noise_and_dropouts_repeat_with_the_seed():
    cfg = SimNoise(seed=3, vision=RsimGaussianNoise(0.01, 0.01, 1), vanishing=0.2)
    runs = []
    for seed in (3, 3, 4):
        env = _env(SimNoise(**{**cfg.__dict__, "seed": seed}))
        try:
            obs = []
            for _ in range(20):
                _step_forward(env)
                obs.append(_observed(env))
            runs.append(obs)
        finally:
            env.close()
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]


def test_a_delayed_robot_gets_its_first_command_latency_ticks_late():
    env = _env(SimNoise(latency_s=(3 * TICK, 3 * TICK)))
    try:
        sent = []
        for _ in range(5):
            _step_forward(env)
            sent.append(env.sent_commands[0].v_x)
        assert sent == [0, 0, 0, 1, 1]
    finally:
        env.close()


def test_without_sim_noise_kicks_are_the_nominal_speed():
    env = _env()
    try:
        cmd = tuple(np.array([0, 0, 0, 1, 0, 0]) for _ in range(2))
        env.step({"team_blue": cmd, "team_yellow": cmd})
        assert {c.kick_v_x for c in env.sent_commands} == {5}
    finally:
        env.close()
