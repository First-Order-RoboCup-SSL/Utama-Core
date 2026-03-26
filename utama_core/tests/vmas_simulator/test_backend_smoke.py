"""Smoke tests for the VMAS SSL backend."""

import math

import torch

from utama_core.rsoccer_simulator.src.Entities import Ball, Frame, Robot
from utama_core.vmas_simulator.env_factory import make_ssl_env
from utama_core.vmas_simulator.src.Scenario.ssl_scenario import SSLScenario
from utama_core.vmas_simulator.src.Simulators.vmas_ssl import VmasSSL
from utama_core.vmas_simulator.src.Utils.config import SSLScenarioConfig


def _make_ssl_scenario() -> SSLScenario:
    scenario = SSLScenario()
    world = scenario.make_world(
        batch_dim=1,
        device=torch.device("cpu"),
        scenario_config=SSLScenarioConfig(n_blue=1, n_yellow=0, max_steps=100),
    )
    scenario._world = world
    scenario.reset_world_at(env_index=None)
    return scenario


def test_vmas_ssl_backend_smoke():
    sim = VmasSSL(
        n_robots_blue=1,
        n_robots_yellow=1,
        num_envs=1,
        device="cpu",
        scenario_config=SSLScenarioConfig(n_blue=1, n_yellow=1),
    )
    try:
        initial = sim.get_frame()
        assert len(initial.robots_blue) == 1
        assert len(initial.robots_yellow) == 1

        sim.send_commands(
            [
                Robot(yellow=False, id=0, v_x=1.0, v_y=0.0, v_theta=0.0),
                Robot(yellow=True, id=0, v_x=0.0, v_y=0.0, v_theta=0.0),
            ]
        )
        updated = sim.get_frame()
        assert updated.robots_blue[0].v_x >= 0.0
        assert updated.ball is not None
    finally:
        sim.stop()


def test_vmas_ssl_backend_does_not_double_integrate_rotation():
    sim = VmasSSL(
        n_robots_blue=1,
        n_robots_yellow=0,
        num_envs=1,
        device="cpu",
        scenario_config=SSLScenarioConfig(n_blue=1, n_yellow=0),
    )
    try:
        before = sim.get_frame().robots_blue[0].theta
        sim.send_commands([Robot(yellow=False, id=0, v_x=0.0, v_y=0.0, v_theta=60.0)])
        after = sim.get_frame().robots_blue[0].theta

        dc = sim.scenario.cfg.dynamics
        expected_delta = math.degrees(dc.robot_max_angular_acceleration * sim.world.dt * sim.world.dt)
        assert math.isclose(after - before, expected_delta, rel_tol=1e-3, abs_tol=1e-3)
    finally:
        sim.stop()


def test_ssl_scenario_goal_reward_is_emitted_on_goal_step():
    env = make_ssl_env(
        num_envs=1,
        device="cpu",
        scenario_config=SSLScenarioConfig(n_blue=1, n_yellow=0, max_steps=100),
    )
    scenario = env.scenario
    fc = scenario.cfg.field_config

    env.reset()
    scenario.ball.set_pos(torch.tensor([[fc.half_length + 0.05, 0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_vel(torch.zeros(1, 2), batch_index=None)

    _, rewards, dones, _ = env.step([torch.zeros(1, 5)])

    assert rewards[0].item() >= scenario.cfg.rewards.goal_scored
    assert dones.item()


def test_ssl_scenario_kick_uses_configured_speed():
    env = make_ssl_env(
        num_envs=1,
        device="cpu",
        scenario_config=SSLScenarioConfig(n_blue=1, n_yellow=0, max_steps=100),
    )
    scenario = env.scenario

    env.reset()
    robot = scenario.blue_agents[0]
    scenario.ball.set_pos(robot.state.pos + torch.tensor([[0.1, 0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_vel(torch.zeros(1, 2), batch_index=None)
    scenario.confirmed_holder[:] = 0

    env.step([torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)])

    assert scenario.ball.state.vel[0, 0].item() > 2.5


def test_ssl_scenario_has_ball_confirms_after_stable_contact():
    scenario = _make_ssl_scenario()
    robot = scenario.blue_agents[0]
    robot.set_pos(torch.tensor([[0.0, 0.0]], dtype=torch.float32), batch_index=None)
    robot.set_rot(torch.tensor([[0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_pos(torch.tensor([[0.08, 0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_vel(torch.zeros(1, 2), batch_index=None)

    scenario.post_step()
    assert scenario.observation(robot)[0, 4].item() == 0.0

    scenario.post_step()
    assert scenario.observation(robot)[0, 4].item() == 1.0


def test_ssl_scenario_precontact_gap_does_not_report_has_ball():
    scenario = _make_ssl_scenario()
    robot = scenario.blue_agents[0]
    contact_dist = scenario.cfg.field_config.robot_radius + scenario.cfg.field_config.ball_radius
    robot.set_pos(torch.tensor([[0.0, 0.0]], dtype=torch.float32), batch_index=None)
    robot.set_rot(torch.tensor([[0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_pos(torch.tensor([[contact_dist + 0.003, 0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_vel(torch.zeros(1, 2), batch_index=None)

    scenario.post_step()
    scenario.post_step()

    assert scenario.observation(robot)[0, 4].item() == 0.0


def test_ssl_scenario_kick_and_dribble_require_confirmed_possession():
    scenario = _make_ssl_scenario()
    robot = scenario.blue_agents[0]
    robot.set_pos(torch.tensor([[0.0, 0.0]], dtype=torch.float32), batch_index=None)
    robot.set_rot(torch.tensor([[0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_pos(torch.tensor([[0.08, 0.0]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_vel(torch.zeros(1, 2), batch_index=None)

    robot.action.u = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    assert not scenario.process_kick(robot).item()
    assert not scenario.compute_dribble_state(robot).item()

    scenario.post_step()
    scenario.post_step()

    robot.action.u = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    assert scenario.compute_dribble_state(robot).item()


def test_vmas_ssl_wrapper_infrared_uses_confirmed_has_ball():
    sim = VmasSSL(
        n_robots_blue=1,
        n_robots_yellow=0,
        num_envs=1,
        device="cpu",
        scenario_config=SSLScenarioConfig(n_blue=1, n_yellow=0),
    )
    try:
        frame = sim.get_frame()
        frame.robots_blue[0] = Robot(yellow=False, id=0, x=0.0, y=0.0, theta=0.0)
        frame.ball = Ball(x=0.08, y=0.0, z=0.0, v_x=0.0, v_y=0.0)
        sim.reset(frame)

        sim.send_commands([Robot(yellow=False, id=0, v_x=0.0, v_y=0.0, v_theta=0.0)])
        assert not sim.get_frame().robots_blue[0].infrared

        sim.send_commands([Robot(yellow=False, id=0, v_x=0.0, v_y=0.0, v_theta=0.0)])
        assert sim.get_frame().robots_blue[0].infrared
    finally:
        sim.stop()
