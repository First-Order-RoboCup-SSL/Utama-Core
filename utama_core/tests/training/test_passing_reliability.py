"""Reliability tests for the passing scenario."""

import math

import torch
from tensordict import TensorDict

from utama_core.training.scenario.passing_config import (
    PassingDynamicsConfig,
    PassingResetRandomizationConfig,
    PassingScenarioConfig,
)
from utama_core.training.scenario.passing_scenario import PassingScenario
from utama_core.training.task import SSLTask
from utama_core.vmas_simulator.src.Utils.config import SSLDynamicsConfig


def _make_scenario(batch_dim: int = 1) -> PassingScenario:
    scenario = PassingScenario()
    cfg = PassingScenarioConfig(
        n_attackers=2,
        n_defenders=1,
        dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=True),
        reset_randomization=PassingResetRandomizationConfig(benchmark_reset=True),
    )
    world = scenario.make_world(
        batch_dim=batch_dim,
        device=torch.device("cpu"),
        scenario_config=cfg,
    )
    scenario._world = world
    scenario.reset_world_at(env_index=None)
    return scenario


def _set_robot(agent, x: float, y: float, theta: float) -> None:
    agent.set_pos(torch.tensor([[x, y]], dtype=torch.float32), batch_index=None)
    agent.set_vel(torch.zeros(1, 2), batch_index=None)
    agent.set_rot(torch.tensor([[theta]], dtype=torch.float32), batch_index=None)
    if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
        agent.state.ang_vel = torch.zeros(1, 1)


def _set_ball(scenario: PassingScenario, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
    scenario.ball.set_pos(torch.tensor([[x, y]], dtype=torch.float32), batch_index=None)
    scenario.ball.set_vel(torch.tensor([[vx, vy]], dtype=torch.float32), batch_index=None)


def _confirm_holder(scenario: PassingScenario, steps: int | None = None) -> None:
    repeats = steps or scenario.cfg.dynamics.pass_confirm_frames
    for _ in range(repeats):
        scenario._update_possession_and_pass_state()


def _step_scenario(
    scenario: PassingScenario,
    actions: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    actions = actions or {}
    for agent in scenario.attackers + scenario.defenders:
        default = torch.zeros(1, agent.action_size)
        agent.action.u = actions.get(agent.name, default)
    for agent in scenario.attackers + scenario.defenders:
        scenario.process_action(agent)
    scenario.world.step()
    scenario.post_step()
    rewards = {agent.name: scenario.reward(agent).clone() for agent in scenario.attackers + scenario.defenders}
    done = scenario.done().clone()
    return rewards, done


def _snapshot(scenario: PassingScenario) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    agent = scenario.attackers[0]
    ang = (
        agent.state.ang_vel.clone()
        if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None
        else torch.zeros(1, 1)
    )
    return agent.state.pos.clone(), scenario.ball.state.pos.clone(), ang


class TestRewardOrder:
    def test_reward_calls_are_order_independent(self):
        scenario = _make_scenario()
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_robot(scenario.attackers[1], 1.0, 0.0, math.pi)
        _set_ball(scenario, 0.25, 0.0)
        scenario.confirmed_holder[:] = 0
        scenario.last_attacker_holder[:] = 0
        scenario.previous_metrics = scenario._clone_metrics(scenario._compute_metrics())
        _set_ball(scenario, 0.55, 0.0)
        scenario.current_metrics = scenario._compute_metrics()

        r01 = (
            scenario.reward(scenario.attackers[0]).clone(),
            scenario.reward(scenario.attackers[1]).clone(),
        )
        state01 = (scenario.pass_count.clone(), scenario.pass_completed.clone())

        scenario2 = _make_scenario()
        _set_robot(scenario2.attackers[0], 0.0, 0.0, 0.0)
        _set_robot(scenario2.attackers[1], 1.0, 0.0, math.pi)
        _set_ball(scenario2, 0.25, 0.0)
        scenario2.confirmed_holder[:] = 0
        scenario2.last_attacker_holder[:] = 0
        scenario2.previous_metrics = scenario2._clone_metrics(scenario2._compute_metrics())
        _set_ball(scenario2, 0.55, 0.0)
        scenario2.current_metrics = scenario2._compute_metrics()

        r10 = (
            scenario2.reward(scenario2.attackers[1]).clone(),
            scenario2.reward(scenario2.attackers[0]).clone(),
        )
        state10 = (scenario2.pass_count.clone(), scenario2.pass_completed.clone())

        assert torch.allclose(r01[0], r10[1])
        assert torch.allclose(r01[1], r10[0])
        assert torch.equal(state01[0], state10[0])
        assert torch.equal(state01[1], state10[1])

    def test_passer_phase2_requires_live_possession_or_active_pass(self):
        scenario = _make_scenario()
        scenario.confirmed_holder[:] = -1
        scenario.last_attacker_holder[:] = 0
        scenario.pass_active[:] = False

        scenario.previous_metrics["passer_to_ball_dist"][:] = 0.4
        scenario.current_metrics["passer_to_ball_dist"][:] = 0.6
        scenario.previous_metrics["passer_capture_error"][:] = 0.4
        scenario.current_metrics["passer_capture_error"][:] = 0.6
        scenario.previous_metrics["ball_to_receiver_dist"][:] = 2.0
        scenario.current_metrics["ball_to_receiver_dist"][:] = 1.0
        scenario.previous_metrics["passer_facing_receiver_cos"][:] = 0.0
        scenario.current_metrics["passer_facing_receiver_cos"][:] = 0.0

        reward = scenario.reward(scenario.attackers[0]).clone()
        info = scenario.info(scenario.attackers[0])

        assert info["reward/pass_delta"].item() == 0.0
        assert info["reward/approach_delta"].item() < 0.0
        assert reward.item() < 0.0

    def test_passer_capture_reward_penalizes_driving_past_ball(self):
        scenario = _make_scenario()
        contact_dist = scenario.cfg.field.robot_radius + scenario.cfg.field.ball_radius
        ball_x = 0.2

        _set_ball(scenario, ball_x, 0.0)
        _set_robot(scenario.attackers[0], ball_x - contact_dist, 0.0, 0.0)
        scenario.previous_metrics = scenario._clone_metrics(scenario._compute_metrics())

        _set_robot(scenario.attackers[0], 0.16, 0.0, 0.0)
        scenario.current_metrics = scenario._compute_metrics()

        reward = scenario.reward(scenario.attackers[0]).clone()
        info = scenario.info(scenario.attackers[0])

        assert info["reward/approach_delta"].item() < 0.0
        assert reward.item() < 0.0

    def test_pre_possession_passer_does_not_get_face_receiver_reward(self):
        scenario = _make_scenario()
        scenario.confirmed_holder[:] = -1
        scenario.pass_active[:] = False
        scenario.previous_metrics["passer_facing_ball_cos"][:] = 0.0
        scenario.current_metrics["passer_facing_ball_cos"][:] = 0.0
        scenario.previous_metrics["passer_facing_receiver_cos"][:] = 0.0
        scenario.current_metrics["passer_facing_receiver_cos"][:] = 1.0

        reward = scenario.reward(scenario.attackers[0]).clone()
        info = scenario.info(scenario.attackers[0])

        assert info["reward/face_receiver"].item() == 0.0
        assert reward.item() == 0.0

    def test_near_ball_overshoot_and_orbit_penalties_apply_before_possession(self):
        scenario = _make_scenario()
        scenario.confirmed_holder[:] = -1
        scenario.pass_active[:] = False
        scenario.previous_metrics["passer_capture_error"][:] = 0.05
        scenario.current_metrics["passer_capture_error"][:] = 0.05
        scenario.previous_metrics["passer_ball_radial_speed"][:] = 0.0
        scenario.current_metrics["passer_ball_radial_speed"][:] = -1.2
        scenario.previous_metrics["passer_ball_tangential_speed"][:] = 0.0
        scenario.current_metrics["passer_ball_tangential_speed"][:] = 0.6

        reward = scenario.reward(scenario.attackers[0]).clone()
        info = scenario.info(scenario.attackers[0])

        assert info["reward/overshoot_speed"].item() < 0.0
        assert info["reward/orbit_speed"].item() < 0.0
        assert reward.item() < 0.0

    def test_acquire_ball_reward_fires_once_on_possession_transition(self):
        scenario = _make_scenario()
        scenario.previous_metrics["passer_has_ball"][:] = 0.0
        scenario.current_metrics["passer_has_ball"][:] = 1.0
        scenario.previous_metrics["passer_facing_ball_cos"][:] = 0.0
        scenario.current_metrics["passer_facing_ball_cos"][:] = 0.0
        scenario.previous_metrics["passer_facing_receiver_cos"][:] = 0.0
        scenario.current_metrics["passer_facing_receiver_cos"][:] = 0.0
        scenario.previous_metrics["ball_to_receiver_dist"][:] = 0.0
        scenario.current_metrics["ball_to_receiver_dist"][:] = 0.0
        scenario.previous_metrics["passer_to_ball_dist"][:] = 0.0
        scenario.current_metrics["passer_to_ball_dist"][:] = 0.0
        scenario.previous_metrics["passer_capture_error"][:] = 0.0
        scenario.current_metrics["passer_capture_error"][:] = 0.0

        reward = scenario.reward(scenario.attackers[0]).clone()
        info = scenario.info(scenario.attackers[0])

        assert info["reward/acquire_ball"].item() == scenario.cfg.rewards.acquire_ball_reward
        assert reward.item() >= scenario.cfg.rewards.acquire_ball_reward

    def test_receiver_approach_reward_waits_for_released_pass(self):
        scenario = _make_scenario()
        receiver = scenario.attackers[1]

        scenario.pass_active[:] = False
        scenario.previous_metrics["receiver_to_ball_dist"][:] = 3.0
        scenario.current_metrics["receiver_to_ball_dist"][:] = 2.0
        scenario.previous_metrics["receiver_capture_error"][:] = 3.0
        scenario.current_metrics["receiver_capture_error"][:] = 2.0
        scenario.previous_metrics["receiver_facing_ball_cos"][:] = 0.0
        scenario.current_metrics["receiver_facing_ball_cos"][:] = 0.0

        reward = scenario.reward(receiver).clone()
        info = scenario.info(receiver)

        assert info["reward/recv_approach"].item() == 0.0
        assert reward.item() == 0.0

        scenario.pass_active[:] = True
        reward = scenario.reward(receiver).clone()
        info = scenario.info(receiver)

        assert info["reward/recv_approach"].item() > 0.0
        assert reward.item() > 0.0


class TestPassDetection:
    def test_authoritative_has_ball_stays_false_until_confirmation(self):
        scenario = _make_scenario()
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_ball(scenario, 0.08, 0.0)

        scenario._update_possession_and_pass_state()
        assert not scenario._has_ball(scenario.attackers[0]).item()

        scenario._update_possession_and_pass_state()
        assert scenario._has_ball(scenario.attackers[0]).item()

    def test_precontact_gap_does_not_count_as_possession(self):
        scenario = _make_scenario()
        contact_dist = scenario.cfg.field.robot_radius + scenario.cfg.field.ball_radius
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_ball(scenario, contact_dist + 0.003, 0.0)

        _confirm_holder(scenario)

        assert not scenario._is_ball_control_candidate(scenario.attackers[0]).item()
        assert not scenario._has_ball(scenario.attackers[0]).item()

    def test_candidate_contact_does_not_enable_preconfirmation_ball_control(self):
        scenario = _make_scenario()
        passer = scenario.attackers[0]
        _set_robot(passer, 0.0, 0.0, 0.0)
        _set_ball(scenario, 0.08, 0.0)

        _step_scenario(
            scenario,
            {
                "attacker_0": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
                "attacker_1": torch.tensor([[1.5, -2.5, 0.0, -1.0]], dtype=torch.float32),
                "defender_0": torch.zeros(1, 4),
            },
        )
        assert not scenario._has_ball(passer).item()
        assert not scenario.kick_fired["attacker_0"].item()
        contact_dist = scenario.cfg.field.robot_radius + scenario.cfg.field.ball_radius
        assert scenario.ball.state.pos[0, 0].item() >= contact_dist - 1e-5

    def test_true_pass_requires_release_and_receiver_confirmation(self):
        scenario = _make_scenario()
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_robot(scenario.attackers[1], 1.0, 0.0, math.pi)
        _set_ball(scenario, 0.08, 0.0)
        _confirm_holder(scenario)
        assert scenario.confirmed_holder.item() == 0

        _set_ball(scenario, 0.30, 0.0)
        _confirm_holder(scenario)
        assert scenario.confirmed_holder.item() == -1
        assert scenario.pass_active.item()

        _set_ball(scenario, 0.92, 0.0)
        _confirm_holder(scenario)
        assert scenario.confirmed_holder.item() == 1
        assert scenario.pass_completed.item()
        assert scenario.pass_count.item() == 1

    def test_self_recontact_does_not_count_as_pass(self):
        scenario = _make_scenario()
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_robot(scenario.attackers[1], 1.0, 0.0, math.pi)
        _set_ball(scenario, 0.08, 0.0)
        _confirm_holder(scenario)

        _set_ball(scenario, 0.30, 0.0)
        _confirm_holder(scenario)

        _set_ball(scenario, 0.08, 0.0)
        _confirm_holder(scenario)
        assert scenario.confirmed_holder.item() == 0
        assert not scenario.pass_completed.item()
        assert scenario.pass_count.item() == 0

    def test_ball_behind_receiver_does_not_trigger_possession(self):
        scenario = _make_scenario()
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_robot(scenario.attackers[1], 1.0, 0.0, 0.0)
        _set_ball(scenario, 0.08, 0.0)
        _confirm_holder(scenario)

        _set_ball(scenario, 0.30, 0.0)
        _confirm_holder(scenario)

        _set_ball(scenario, 0.92, 0.0)
        _confirm_holder(scenario)
        assert scenario.confirmed_holder.item() == -1
        assert not scenario.pass_completed.item()

    def test_holder_chatter_does_not_flip_without_confirmation(self):
        scenario = _make_scenario()
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_robot(scenario.attackers[1], 1.0, 0.0, math.pi)
        _set_ball(scenario, 0.08, 0.0)
        _confirm_holder(scenario)
        assert scenario.confirmed_holder.item() == 0

        _set_ball(scenario, 0.30, 0.0)
        scenario._update_possession_and_pass_state()
        _set_ball(scenario, 0.92, 0.0)
        scenario._update_possession_and_pass_state()
        assert scenario.confirmed_holder.item() == 0


class TestDeterminismAndPhysics:
    def test_post_step_dribble_lock_keeps_ball_at_kicker_face(self):
        scenario = _make_scenario()
        passer = scenario.attackers[0]
        _set_robot(passer, 0.0, 0.0, 0.0)
        _set_ball(scenario, 0.02, 0.0)
        scenario.confirmed_holder[:] = 0
        scenario.kick_fired[passer.name][:] = False
        scenario._step_requests[passer.name] = {
            "kick_request": torch.zeros(1, dtype=torch.bool),
            "dribble_request": torch.ones(1, dtype=torch.bool),
            "kick_alignment": torch.zeros(1),
        }

        scenario._apply_ball_effects_post_step()

        contact_dist = scenario.cfg.field.robot_radius + scenario.cfg.field.ball_radius
        assert torch.allclose(
            scenario.ball.state.pos,
            torch.tensor([[contact_dist, 0.0]], dtype=torch.float32),
            atol=1e-6,
        )

    def test_scripted_go_to_ball_rollout_reaches_confirmed_possession(self):
        task = SSLTask.from_name("ssl_2v0_unified")
        env = task.get_env_fun(num_envs=1, continuous_actions=True, seed=0, device="cpu")()
        try:
            env.reset()
            scenario = env._env.scenario

            for _ in range(180):
                scenario = env._env.scenario
                ball_pos = scenario.ball.state.pos[0]
                passer = scenario.attackers[0]
                receiver = scenario.attackers[1]
                to_ball = ball_pos - passer.state.pos[0]
                target_oren = math.atan2(to_ball[1].item(), to_ball[0].item())
                actions = TensorDict(
                    {
                        "passer": TensorDict(
                            {
                                "action": torch.tensor(
                                    [
                                        [
                                            [
                                                ball_pos[0].item() / scenario.cfg.field.half_length,
                                                ball_pos[1].item() / scenario.cfg.field.half_width,
                                                target_oren / math.pi,
                                                -1.0,
                                            ]
                                        ]
                                    ],
                                    dtype=torch.float32,
                                )
                            },
                            batch_size=[1, 1],
                        ),
                        "receiver": TensorDict(
                            {
                                "action": torch.tensor(
                                    [
                                        [
                                            [
                                                receiver.state.pos[0, 0].item() / scenario.cfg.field.half_length,
                                                receiver.state.pos[0, 1].item() / scenario.cfg.field.half_width,
                                                receiver.state.rot[0, 0].item() / math.pi,
                                                -1.0,
                                            ]
                                        ]
                                    ],
                                    dtype=torch.float32,
                                )
                            },
                            batch_size=[1, 1],
                        ),
                    },
                    batch_size=[1],
                )
                env.step(actions)
                if env._env.scenario._has_ball(env._env.scenario.attackers[0]).item():
                    break

            assert env._env.scenario._has_ball(env._env.scenario.attackers[0]).item()
        finally:
            env.close()

    def test_rollout_is_deterministic(self):
        scenario_a = _make_scenario()
        scenario_b = _make_scenario()
        actions = {
            "attacker_0": torch.tensor([[2.0, 0.0, 0.0, -1.0]]),
            "attacker_1": torch.tensor([[1.0, -1.0, math.pi, -1.0]]),
            "defender_0": torch.zeros(1, 4),
        }

        for _ in range(5):
            rewards_a, done_a = _step_scenario(scenario_a, actions)
            rewards_b, done_b = _step_scenario(scenario_b, actions)
            assert torch.equal(done_a, done_b)
            for name in rewards_a:
                assert torch.allclose(rewards_a[name], rewards_b[name])
            assert all(torch.allclose(a, b) for a, b in zip(_snapshot(scenario_a), _snapshot(scenario_b)))

    def test_robot_speed_and_angular_velocity_respect_limits(self):
        scenario = _make_scenario()
        actions = {
            "attacker_0": torch.tensor([[4.0, 2.0, math.pi, -1.0]]),
            "attacker_1": torch.tensor([[4.0, -2.0, -math.pi / 2, -1.0]]),
        }
        for _ in range(15):
            _step_scenario(scenario, actions)

        for agent in scenario.attackers:
            speed = torch.norm(agent.state.vel, dim=-1)
            assert (speed <= scenario.cfg.dynamics.robot_max_speed + 1e-5).all()
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
                assert (agent.state.ang_vel.abs() <= scenario.cfg.dynamics.robot_max_angular_vel + 1e-5).all()

    def test_kick_cooldown_blocks_immediate_refire(self):
        scenario = _make_scenario()
        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_ball(scenario, 0.08, 0.0)
        _confirm_holder(scenario)
        action = {"attacker_0": torch.tensor([[4.0, 0.0, 0.0, 1.0]])}
        _step_scenario(scenario, action)
        first_kick = scenario.kick_fired["attacker_0"].clone()

        _set_robot(scenario.attackers[0], 0.0, 0.0, 0.0)
        _set_ball(scenario, 0.08, 0.0)
        scenario.confirmed_holder[:] = 0
        _step_scenario(scenario, action)
        second_kick = scenario.kick_fired["attacker_0"].clone()

        assert first_kick.item()
        assert not second_kick.item()

    def test_ball_slows_down_monotonically_when_untouched(self):
        scenario = _make_scenario()
        _set_ball(scenario, 0.0, 0.0, vx=1.0, vy=0.0)
        speeds = []
        for _ in range(6):
            _step_scenario(scenario, {})
            speeds.append(torch.norm(scenario.ball.state.vel, dim=-1).item())
        assert speeds == sorted(speeds, reverse=True)

    def test_ball_stays_inside_world_bounds(self):
        scenario = _make_scenario()
        _set_ball(
            scenario,
            scenario.cfg.field.half_length - 0.05,
            0.0,
            vx=2.5,
            vy=0.0,
        )
        for _ in range(20):
            _step_scenario(scenario, {})
        limit = scenario.cfg.field.half_length + scenario.cfg.field.boundary_margin + 1e-4
        assert scenario.ball.state.pos[:, 0].abs().max().item() <= limit


class TestReferenceAlignment:
    def test_common_physics_defaults_match_ssl_reference_backend(self):
        passing = PassingDynamicsConfig()
        reference = SSLDynamicsConfig()
        assert passing.dt == reference.dt
        assert passing.substeps == reference.substeps
        assert passing.robot_mass == reference.robot_mass
        assert passing.ball_mass == reference.ball_mass
        assert passing.ball_friction == reference.ball_friction
        assert passing.robot_friction == reference.robot_friction
        assert passing.kick_speed == reference.kick_speed
        assert passing.robot_max_acceleration == reference.robot_max_acceleration
        assert passing.robot_max_angular_acceleration == reference.robot_max_angular_acceleration
