"""VMAS scenario for SSL passing drills."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from vmas.simulator.scenario import BaseScenario

from utama_core.training.scenario.batched_pid import (
    BatchedOrientationPID,
    BatchedTranslationPID,
)
from utama_core.training.scenario.passing_config import (
    MacroAction,
    PassingScenarioConfig,
)
from utama_core.training.scenario.passing_rewards import compute_passing_reward
from utama_core.training.scenario.velocity_holonomic import VelocityHolonomic


class PassingScenario(BaseScenario):
    """Passing drill scenario with force-based and legacy kinematic modes."""

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.cfg: PassingScenarioConfig = kwargs.get("scenario_config", PassingScenarioConfig())
        fc = self.cfg.field
        dc = self.cfg.dynamics

        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=dc.dt,
            substeps=dc.substeps,
            x_semidim=fc.half_length + fc.boundary_margin,
            y_semidim=fc.half_width + fc.boundary_margin,
            drag=0.0,
            linear_friction=0.0,
            angular_friction=0.0,
            collision_force=dc.collision_force,
        )

        if dc.use_unified_actions:
            action_size = 4
            u_multiplier = [fc.half_length, fc.half_width, math.pi, 1.0]
        elif dc.use_macro_actions:
            action_size = 3
            u_multiplier = [1.0, fc.half_length, fc.half_width]
        else:
            action_size = 6
            u_multiplier = [
                dc.action_delta_range,
                dc.action_delta_range,
                math.pi,
                1.0,
                1.0,
                1.0,
            ]

        max_force = dc.robot_mass * dc.robot_max_acceleration
        max_torque = dc.robot_mass * (fc.robot_radius**2) * dc.robot_max_angular_acceleration

        self.attackers: list[Agent] = []
        for i in range(self.cfg.n_attackers):
            agent = Agent(
                name=f"attacker_{i}",
                shape=Sphere(radius=fc.robot_radius),
                movable=True,
                rotatable=True,
                collide=True,
                mass=dc.robot_mass,
                max_speed=dc.robot_max_speed,
                max_f=max_force if dc.physics_mode == "force_based" else None,
                max_t=max_torque if dc.physics_mode == "force_based" else None,
                color=torch.tensor([0.2, 0.2, 0.9]),
                dynamics=self._make_agent_dynamics(),
                action_size=action_size,
                u_multiplier=u_multiplier,
                drag=0.0,
                linear_friction=dc.robot_friction,
            )
            world.add_agent(agent)
            self.attackers.append(agent)

        self.defenders: list[Agent] = []
        for i in range(self.cfg.n_defenders):
            agent = Agent(
                name=f"defender_{i}",
                shape=Sphere(radius=fc.robot_radius),
                movable=True,
                rotatable=True,
                collide=True,
                mass=dc.robot_mass,
                max_speed=dc.robot_max_speed,
                max_f=max_force if dc.physics_mode == "force_based" else None,
                max_t=max_torque if dc.physics_mode == "force_based" else None,
                color=torch.tensor([0.9, 0.9, 0.2]),
                dynamics=self._make_agent_dynamics(),
                action_size=action_size,
                u_multiplier=u_multiplier,
                drag=0.0,
                linear_friction=dc.robot_friction,
            )
            world.add_agent(agent)
            self.defenders.append(agent)

        self.ball = Landmark(
            name="ball",
            shape=Sphere(radius=fc.ball_radius),
            movable=True,
            rotatable=False,
            collide=True,
            mass=dc.ball_mass,
            color=torch.tensor([1.0, 0.5, 0.0]),
            linear_friction=dc.ball_friction,
        )
        world.add_landmark(self.ball)
        self._create_walls(world)

        self.steps = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.global_frame = 0
        self.pass_completed = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.ball_intercepted = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.pass_count = torch.zeros(batch_dim, device=device, dtype=torch.long)

        self.confirmed_holder = torch.full((batch_dim,), -1, device=device, dtype=torch.long)
        self.candidate_holder = torch.full((batch_dim,), -1, device=device, dtype=torch.long)
        self.candidate_frames = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.holder_switch_cooldown = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.pass_active = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.pass_start_ball_pos = torch.zeros(batch_dim, 2, device=device)
        self.last_attacker_holder = torch.full((batch_dim,), -1, device=device, dtype=torch.long)

        self.current_metrics = self._empty_metrics(batch_dim, device)
        self.previous_metrics = self._empty_metrics(batch_dim, device)
        self._needs_metric_commit = False

        self.prev_defender_vel: dict[str, Tensor] = {
            d.name: torch.zeros(batch_dim, 2, device=device) for d in self.defenders
        }
        self.prev_defender_rot: dict[str, Tensor] = {
            d.name: torch.zeros(batch_dim, 1, device=device) for d in self.defenders
        }
        self.kick_cooldown: dict[str, Tensor] = {
            a.name: torch.zeros(batch_dim, device=device, dtype=torch.long) for a in self.attackers + self.defenders
        }
        self.kick_fired: dict[str, Tensor] = {
            a.name: torch.zeros(batch_dim, device=device, dtype=torch.bool) for a in self.attackers + self.defenders
        }
        self.current_macro_action: dict[str, Tensor] = {
            a.name: torch.zeros(batch_dim, device=device, dtype=torch.long) for a in self.attackers + self.defenders
        }
        self._reward_components: dict[str, dict[str, Tensor]] = {}
        self._step_requests: dict[str, dict[str, Tensor]] = {}
        self._agents_processed_this_step = 0

        self.pid_trans = BatchedTranslationPID()
        self.pid_oren = BatchedOrientationPID()
        self._agent_names = [a.name for a in self.attackers + self.defenders]
        self._holder_name_to_id = {name: idx for idx, name in enumerate(self._agent_names)}

        return world

    def _make_agent_dynamics(self):
        dc = self.cfg.dynamics
        if dc.physics_mode == "force_based":
            return HolonomicWithRotation()
        return VelocityHolonomic(
            max_speed=dc.robot_max_speed,
            max_angular_vel=dc.robot_max_angular_vel,
        )

    def _create_walls(self, world: World) -> None:
        fc = self.cfg.field
        wall_thickness = 0.05
        self.walls: list[tuple[str, Landmark, float, float]] = []
        for name, y_sign in [("top", 1), ("bottom", -1)]:
            wall = Landmark(
                name=f"wall_{name}",
                shape=Box(length=2 * (fc.half_length + fc.boundary_margin), width=wall_thickness),
                movable=False,
                collide=True,
                color=torch.tensor([0.3, 0.6, 0.3]),
            )
            world.add_landmark(wall)
            self.walls.append((name, wall, 0.0, y_sign * fc.half_width))

        for name, x_sign in [("left", -1), ("right", 1)]:
            wall = Landmark(
                name=f"wall_{name}",
                shape=Box(length=wall_thickness, width=2 * (fc.half_width + fc.boundary_margin)),
                movable=False,
                collide=True,
                color=torch.tensor([0.3, 0.6, 0.3]),
            )
            world.add_landmark(wall)
            self.walls.append((name, wall, x_sign * fc.half_length, 0.0))

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            self._reset_world_batch()
            self.global_frame = 0
            self._reward_components.clear()
            self._step_requests.clear()
            self._agents_processed_this_step = 0
            self._needs_metric_commit = False
            return

        fc = self.cfg.field
        rcfg = self.cfg.reset_randomization
        device = self.world.device

        if rcfg.benchmark_reset:
            ball_pos = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32)
        else:
            ball_pos = torch.stack(
                [
                    self._sample_scalar(rcfg.ball_x_range),
                    self._sample_scalar(rcfg.ball_y_range),
                ]
            )
        goal_center = torch.tensor([fc.half_length, 0.0], device=device, dtype=torch.float32)
        self._set_entity_state(self.ball, ball_pos[0].item(), ball_pos[1].item(), env_index)

        if self.cfg.n_attackers >= 1:
            angle = 0.35 if rcfg.benchmark_reset else self._sample_scalar(rcfg.passer_angle_range)
            radius = 1.0 if rcfg.benchmark_reset else self._sample_scalar(rcfg.passer_radius_range)
            angle_val = angle if isinstance(angle, float) else angle.item()
            radius_val = radius if isinstance(radius, float) else radius.item()
            offset = torch.tensor(
                [radius_val * math.cos(angle_val), radius_val * math.sin(angle_val)],
                device=device,
                dtype=torch.float32,
            )
            passer_pos = ball_pos - offset
            self._set_robot_state(self.attackers[0], passer_pos, goal_center - passer_pos, env_index)

        if self.cfg.n_attackers >= 2:
            if rcfg.benchmark_reset:
                receiver_pos = torch.tensor([1.5, -2.5], device=device, dtype=torch.float32)
                receiver_rot = 0.0
            else:
                receiver_pos = torch.stack(
                    [
                        self._sample_scalar(rcfg.receiver_x_range),
                        self._sample_scalar(rcfg.receiver_y_range),
                    ]
                )
                receiver_rot = self._sample_scalar(rcfg.receiver_rot_range).item()
            self._set_robot_state(
                self.attackers[1],
                receiver_pos,
                None,
                env_index,
                explicit_rot=receiver_rot,
            )

        if self.cfg.n_defenders >= 1:
            dist = 0.75 if rcfg.benchmark_reset else self._sample_scalar(rcfg.defender0_distance_range).item()
            vec_to_goal = goal_center - ball_pos
            vec_to_goal = vec_to_goal / torch.norm(vec_to_goal).clamp(min=1e-8)
            defender_pos = ball_pos + vec_to_goal * dist
            self._set_robot_state(self.defenders[0], defender_pos, ball_pos - defender_pos, env_index)

        if self.cfg.n_defenders >= 2:
            if rcfg.benchmark_reset:
                defender_pos = torch.tensor([1.5, -2.0], device=device, dtype=torch.float32)
            else:
                defender_pos = torch.stack(
                    [
                        self._sample_scalar(rcfg.defender1_x_range),
                        self._sample_scalar(rcfg.defender1_y_range),
                    ]
                )
            self._set_robot_state(self.defenders[1], defender_pos, ball_pos - defender_pos, env_index)

        for _, wall, wx, wy in self.walls:
            self._set_entity_state(wall, wx, wy, env_index, movable=False)

        self._reset_tracking(env_index)

    def _reset_world_batch(self) -> None:
        batch_dim = self.world.batch_dim
        fc = self.cfg.field
        rcfg = self.cfg.reset_randomization
        device = self.world.device

        if rcfg.benchmark_reset:
            ball_pos = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32).unsqueeze(0).expand(batch_dim, -1)
        else:
            ball_pos = torch.stack(
                [
                    self._sample_scalars(rcfg.ball_x_range, batch_dim),
                    self._sample_scalars(rcfg.ball_y_range, batch_dim),
                ],
                dim=-1,
            )

        goal_center = torch.tensor([fc.half_length, 0.0], device=device, dtype=torch.float32).unsqueeze(0)
        self._set_entity_state_batch(self.ball, ball_pos)

        if self.cfg.n_attackers >= 1:
            if rcfg.benchmark_reset:
                angle = torch.full((batch_dim,), 0.35, device=device, dtype=torch.float32)
                radius = torch.full((batch_dim,), 1.0, device=device, dtype=torch.float32)
            else:
                angle = self._sample_scalars(rcfg.passer_angle_range, batch_dim)
                radius = self._sample_scalars(rcfg.passer_radius_range, batch_dim)
            offset = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
            passer_pos = ball_pos - offset
            self._set_robot_state_batch(self.attackers[0], passer_pos, goal_center - passer_pos)

        if self.cfg.n_attackers >= 2:
            if rcfg.benchmark_reset:
                receiver_pos = (
                    torch.tensor([1.5, -2.5], device=device, dtype=torch.float32).unsqueeze(0).expand(batch_dim, -1)
                )
                receiver_rot = torch.zeros(batch_dim, 1, device=device, dtype=torch.float32)
            else:
                receiver_pos = torch.stack(
                    [
                        self._sample_scalars(rcfg.receiver_x_range, batch_dim),
                        self._sample_scalars(rcfg.receiver_y_range, batch_dim),
                    ],
                    dim=-1,
                )
                receiver_rot = self._sample_scalars(rcfg.receiver_rot_range, batch_dim).unsqueeze(-1)
            self._set_robot_state_batch(
                self.attackers[1],
                receiver_pos,
                None,
                explicit_rot=receiver_rot,
            )

        if self.cfg.n_defenders >= 1:
            if rcfg.benchmark_reset:
                dist = torch.full((batch_dim, 1), 0.75, device=device, dtype=torch.float32)
            else:
                dist = self._sample_scalars(rcfg.defender0_distance_range, batch_dim).unsqueeze(-1)
            vec_to_goal = goal_center - ball_pos
            vec_to_goal = vec_to_goal / torch.norm(vec_to_goal, dim=-1, keepdim=True).clamp(min=1e-8)
            defender_pos = ball_pos + vec_to_goal * dist
            self._set_robot_state_batch(self.defenders[0], defender_pos, ball_pos - defender_pos)

        if self.cfg.n_defenders >= 2:
            if rcfg.benchmark_reset:
                defender_pos = (
                    torch.tensor([1.5, -2.0], device=device, dtype=torch.float32).unsqueeze(0).expand(batch_dim, -1)
                )
            else:
                defender_pos = torch.stack(
                    [
                        self._sample_scalars(rcfg.defender1_x_range, batch_dim),
                        self._sample_scalars(rcfg.defender1_y_range, batch_dim),
                    ],
                    dim=-1,
                )
            self._set_robot_state_batch(self.defenders[1], defender_pos, ball_pos - defender_pos)

        for _, wall, wx, wy in self.walls:
            self._set_entity_state_batch(wall, torch.tensor([wx, wy], device=device, dtype=torch.float32))

        self._reset_tracking(env_index=None)

    def _sample_scalar(self, bounds: tuple[float, float]) -> Tensor:
        low, high = bounds
        return torch.empty((), device=self.world.device).uniform_(low, high)

    def _sample_scalars(self, bounds: tuple[float, float], count: int) -> Tensor:
        low, high = bounds
        return torch.empty(count, device=self.world.device).uniform_(low, high)

    def _set_robot_state(
        self,
        agent: Agent,
        pos: Tensor,
        facing_vec: Tensor | None,
        env_index: int | None,
        explicit_rot: float | None = None,
    ) -> None:
        rot = explicit_rot
        if rot is None and facing_vec is not None:
            rot = math.atan2(facing_vec[1].item(), facing_vec[0].item())
        if rot is None:
            rot = 0.0

        if env_index is None:
            agent.set_pos(pos.unsqueeze(0).expand(self.world.batch_dim, -1), batch_index=None)
            agent.set_vel(torch.zeros(self.world.batch_dim, 2, device=self.world.device), batch_index=None)
            agent.set_rot(
                torch.full((self.world.batch_dim, 1), rot, device=self.world.device, dtype=torch.float32),
                batch_index=None,
            )
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
                agent.state.ang_vel = torch.zeros(self.world.batch_dim, 1, device=self.world.device)
        else:
            agent.set_pos(pos, batch_index=env_index)
            agent.set_vel(torch.zeros(2, device=self.world.device), batch_index=env_index)
            agent.set_rot(torch.tensor([rot], device=self.world.device), batch_index=env_index)
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
                agent.state.ang_vel[env_index] = 0.0

    def _set_robot_state_batch(
        self,
        agent: Agent,
        pos: Tensor,
        facing_vec: Tensor | None,
        explicit_rot: Tensor | None = None,
    ) -> None:
        if explicit_rot is None:
            if facing_vec is None:
                rot = torch.zeros(self.world.batch_dim, 1, device=self.world.device, dtype=torch.float32)
            else:
                rot = torch.atan2(facing_vec[:, 1], facing_vec[:, 0]).unsqueeze(-1)
        else:
            rot = explicit_rot
            if rot.ndim == 1:
                rot = rot.unsqueeze(-1)

        agent.set_pos(pos, batch_index=None)
        agent.set_vel(torch.zeros(self.world.batch_dim, 2, device=self.world.device), batch_index=None)
        agent.set_rot(rot.to(device=self.world.device, dtype=torch.float32), batch_index=None)
        if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
            agent.state.ang_vel = torch.zeros(self.world.batch_dim, 1, device=self.world.device)

    def _set_entity_state(self, entity, x, y, env_index, movable=True):
        pos = torch.tensor([x, y], device=self.world.device, dtype=torch.float32)
        if env_index is None:
            entity.set_pos(pos.unsqueeze(0).expand(self.world.batch_dim, -1), batch_index=None)
            if movable:
                entity.set_vel(torch.zeros(self.world.batch_dim, 2, device=self.world.device), batch_index=None)
        else:
            entity.set_pos(pos, batch_index=env_index)
            if movable:
                entity.set_vel(torch.zeros(2, device=self.world.device), batch_index=env_index)

    def _set_entity_state_batch(self, entity, pos: Tensor, movable: bool = True):
        if pos.ndim == 1:
            pos = pos.unsqueeze(0).expand(self.world.batch_dim, -1)
        entity.set_pos(pos.to(device=self.world.device, dtype=torch.float32), batch_index=None)
        if movable:
            entity.set_vel(torch.zeros(self.world.batch_dim, 2, device=self.world.device), batch_index=None)

    def _reset_tracking(self, env_index: int | None) -> None:
        batch_dim = self.world.batch_dim
        device = self.world.device
        holder_fill = torch.full((batch_dim,), -1, device=device, dtype=torch.long)
        if env_index is None:
            self.steps.zero_()
            self.global_frame = 0
            self.pass_completed.zero_()
            self.ball_intercepted.zero_()
            self.pass_count.zero_()
            self.confirmed_holder.copy_(holder_fill)
            self.candidate_holder.copy_(holder_fill)
            self.candidate_frames.zero_()
            self.holder_switch_cooldown.zero_()
            self.pass_active.zero_()
            self.pass_start_ball_pos.zero_()
            self.last_attacker_holder.copy_(holder_fill)
            for name in self.kick_cooldown:
                self.kick_cooldown[name].zero_()
                self.kick_fired[name].zero_()
                if name in self.current_macro_action:
                    self.current_macro_action[name].zero_()
            for name in self.prev_defender_vel:
                self.prev_defender_vel[name].zero_()
                self.prev_defender_rot[name].zero_()
            self._reward_components.clear()
            self._step_requests.clear()
            self._agents_processed_this_step = 0
            self._needs_metric_commit = False
            for agent in self.attackers + self.defenders:
                self.pid_trans.reset(agent.name)
                self.pid_oren.reset(agent.name)
        else:
            self.steps[env_index] = 0
            self.pass_completed[env_index] = False
            self.ball_intercepted[env_index] = False
            self.pass_count[env_index] = 0
            self.confirmed_holder[env_index] = -1
            self.candidate_holder[env_index] = -1
            self.candidate_frames[env_index] = 0
            self.holder_switch_cooldown[env_index] = 0
            self.pass_active[env_index] = False
            self.pass_start_ball_pos[env_index] = 0.0
            self.last_attacker_holder[env_index] = -1
            for name in self.kick_cooldown:
                self.kick_cooldown[name][env_index] = 0
                self.kick_fired[name][env_index] = False
                if name in self.current_macro_action:
                    self.current_macro_action[name][env_index] = 0
            for name in self.prev_defender_vel:
                self.prev_defender_vel[name][env_index] = 0.0
                self.prev_defender_rot[name][env_index] = 0.0
            for agent in self.attackers + self.defenders:
                self.pid_trans.reset(agent.name, batch_index=env_index)
                self.pid_oren.reset(agent.name, batch_index=env_index)

        metrics = self._compute_metrics()
        if env_index is None:
            self.current_metrics = self._clone_metrics(metrics)
            self.previous_metrics = self._clone_metrics(metrics)
        else:
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for name, tensor in value.items():
                        self.current_metrics[key][name][env_index] = tensor[env_index]
                        self.previous_metrics[key][name][env_index] = tensor[env_index]
                else:
                    self.current_metrics[key][env_index] = value[env_index]
                    self.previous_metrics[key][env_index] = value[env_index]

    def _empty_metrics(self, batch_dim: int, device: torch.device) -> dict[str, Any]:
        return {
            "passer_to_ball_dist": torch.zeros(batch_dim, device=device),
            "passer_capture_error": torch.zeros(batch_dim, device=device),
            "passer_facing_ball_cos": torch.zeros(batch_dim, device=device),
            "passer_ball_radial_speed": torch.zeros(batch_dim, device=device),
            "passer_ball_tangential_speed": torch.zeros(batch_dim, device=device),
            "passer_has_ball": torch.zeros(batch_dim, device=device),
            "ball_to_receiver_dist": torch.zeros(batch_dim, device=device),
            "receiver_to_ball_dist": torch.zeros(batch_dim, device=device),
            "receiver_capture_error": torch.zeros(batch_dim, device=device),
            "receiver_facing_ball_cos": torch.zeros(batch_dim, device=device),
            "receiver_has_ball": torch.zeros(batch_dim, device=device),
            "passer_facing_receiver_cos": torch.zeros(batch_dim, device=device),
            "defender_to_ball_dist": {
                defender.name: torch.zeros(batch_dim, device=device) for defender in self.defenders
            },
            "ball_speed": torch.zeros(batch_dim, device=device),
        }

    def _clone_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                cloned[key] = {name: tensor.clone() for name, tensor in value.items()}
            else:
                cloned[key] = value.clone()
        return cloned

    def _compute_metrics(self) -> dict[str, Any]:
        metrics = self._empty_metrics(self.world.batch_dim, self.world.device)
        ball_pos = self.ball.state.pos
        metrics["ball_speed"] = torch.norm(self.ball.state.vel, dim=-1)
        if self.cfg.n_attackers >= 1:
            metrics["passer_to_ball_dist"] = torch.norm(self.attackers[0].state.pos - ball_pos, dim=-1)
            metrics["passer_capture_error"] = self._capture_error(self.attackers[0], ball_pos)
            metrics["passer_facing_ball_cos"] = self._facing_cos(self.attackers[0], ball_pos)
            radial_speed, tangential_speed = self._ball_motion_components(self.attackers[0], ball_pos)
            metrics["passer_ball_radial_speed"] = radial_speed
            metrics["passer_ball_tangential_speed"] = tangential_speed
            metrics["passer_has_ball"] = self._has_ball(self.attackers[0]).float()
        if self.cfg.n_attackers >= 2:
            receiver = self.attackers[1]
            recv_ball_dist = torch.norm(ball_pos - receiver.state.pos, dim=-1)
            metrics["ball_to_receiver_dist"] = recv_ball_dist
            metrics["receiver_to_ball_dist"] = recv_ball_dist
            metrics["receiver_capture_error"] = self._capture_error(receiver, ball_pos)
            metrics["receiver_facing_ball_cos"] = self._facing_cos(receiver, ball_pos)
            metrics["receiver_has_ball"] = self._has_ball(receiver).float()
            metrics["passer_facing_receiver_cos"] = self._facing_cos(self.attackers[0], receiver.state.pos)
        for defender in self.defenders:
            metrics["defender_to_ball_dist"][defender.name] = torch.norm(defender.state.pos - ball_pos, dim=-1)
        return metrics

    def _facing_cos(self, agent: Agent, target_pos: Tensor) -> Tensor:
        to_target = target_pos - agent.state.pos
        to_target = to_target / torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-8)
        rot = agent.state.rot.squeeze(-1)
        facing = torch.stack([torch.cos(rot), torch.sin(rot)], dim=-1)
        return (facing * to_target).sum(dim=-1)

    def _contact_point(self, agent: Agent) -> Tensor:
        contact_dist = self.cfg.field.robot_radius + self.cfg.field.ball_radius
        return agent.state.pos + self._agent_facing(agent) * contact_dist

    def _capture_control_targets(
        self,
        agent: Agent,
        requested_target: Tensor,
        requested_oren: Tensor | None = None,
        *,
        force_capture: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        has_ball = self._has_ball(agent)
        to_ball = self.ball.state.pos - agent.state.pos
        to_ball_dist = torch.norm(to_ball, dim=-1, keepdim=True)
        facing = self._agent_facing(agent)
        approach_dir = torch.where(
            to_ball_dist > 1e-6,
            to_ball / to_ball_dist.clamp(min=1e-8),
            facing,
        )
        if force_capture is None:
            target_to_ball_dist = torch.norm(requested_target - self.ball.state.pos, dim=-1)
            close_to_ball = to_ball_dist.squeeze(-1) <= self.cfg.dynamics.capture_lock_radius
            force_capture = (target_to_ball_dist <= self.cfg.dynamics.capture_target_radius) | close_to_ball
        capture_mask = force_capture & ~has_ball

        contact_dist = self.cfg.field.robot_radius + self.cfg.field.ball_radius
        press_dist = max(contact_dist - self.cfg.dynamics.capture_target_overlap, 0.0)
        capture_target = self.ball.state.pos - approach_dir * press_dist
        nav_target = torch.where(capture_mask.unsqueeze(-1), capture_target, requested_target)

        if requested_oren is None:
            return nav_target, None, capture_mask

        capture_oren = torch.atan2(approach_dir[:, 1], approach_dir[:, 0]).unsqueeze(-1)
        orientation_target = torch.where(capture_mask.unsqueeze(-1), capture_oren, requested_oren)
        return nav_target, orientation_target, capture_mask

    def _capture_nav_target(
        self,
        agent: Agent,
        requested_target: Tensor,
        *,
        force_capture: Tensor | None = None,
    ) -> Tensor:
        nav_target, _, _ = self._capture_control_targets(agent, requested_target, force_capture=force_capture)
        return nav_target

    def _capture_error(self, agent: Agent, ball_pos: Tensor | None = None) -> Tensor:
        if ball_pos is None:
            ball_pos = self.ball.state.pos
        return torch.norm(self._contact_point(agent) - ball_pos, dim=-1)

    def _ball_motion_components(self, agent: Agent, ball_pos: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if ball_pos is None:
            ball_pos = self.ball.state.pos
        to_ball = ball_pos - agent.state.pos
        to_ball_dist = torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-8)
        radial_dir = to_ball / to_ball_dist
        radial_speed = (agent.state.vel * radial_dir).sum(dim=-1)
        tangential_vel = agent.state.vel - radial_dir * radial_speed.unsqueeze(-1)
        tangential_speed = torch.norm(tangential_vel, dim=-1)
        return radial_speed, tangential_speed

    def _resolve_ball_penetration(self) -> None:
        contact_dist = self.cfg.field.robot_radius + self.cfg.field.ball_radius
        ball_pos = self.ball.state.pos
        ball_vel = self.ball.state.vel

        for holder_id, agent in enumerate(self.attackers + self.defenders):
            to_ball = ball_pos - agent.state.pos
            dist = torch.norm(to_ball, dim=-1, keepdim=True)
            overlapping = dist.squeeze(-1) < contact_dist
            if not overlapping.any():
                continue

            facing = self._agent_facing(agent)
            normal = torch.where(
                dist > 1e-8,
                to_ball / dist.clamp(min=1e-8),
                facing,
            )
            separated_pos = agent.state.pos + normal * contact_dist
            holder_mask = overlapping & (self.confirmed_holder == holder_id)
            holder_pos = agent.state.pos + facing * contact_dist
            ball_pos = torch.where(
                overlapping.unsqueeze(-1),
                torch.where(holder_mask.unsqueeze(-1), holder_pos, separated_pos),
                ball_pos,
            )

            rel_vel = ball_vel - agent.state.vel
            inward_speed = (rel_vel * normal).sum(dim=-1)
            inward_mask = overlapping & (inward_speed < 0)
            ball_vel = torch.where(
                inward_mask.unsqueeze(-1),
                ball_vel - normal * inward_speed.unsqueeze(-1),
                ball_vel,
            )
            ball_vel = torch.where(holder_mask.unsqueeze(-1), agent.state.vel, ball_vel)

        self.ball.state.pos = ball_pos
        self.ball.state.vel = ball_vel

    def _is_ball_control_candidate(self, agent: Agent) -> Tensor:
        dist = torch.norm(self.ball.state.pos - agent.state.pos, dim=-1)
        contact_dist = self.cfg.field.robot_radius + self.cfg.field.ball_radius
        control_dist = min(self.cfg.dynamics.dribble_dist_threshold, contact_dist)
        return (dist <= control_dist) & (
            self._facing_cos(agent, self.ball.state.pos) >= self.cfg.dynamics.possession_cone_cos
        )

    def _has_ball(self, agent: Agent) -> Tensor:
        holder_id = self._holder_name_to_id.get(agent.name, -1)
        if holder_id < 0:
            return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        return self.confirmed_holder == holder_id

    def process_action(self, agent: Agent):
        self._begin_step_if_needed()
        if agent in self.defenders and self.cfg.defender_behavior == "fixed":
            agent.action.u = torch.zeros_like(agent.action.u)
            self._step_requests[agent.name] = self._empty_request(agent)
            self._mark_agent_processed()
            return

        if self.cfg.dynamics.use_unified_actions:
            self._process_unified_action(agent)
        elif self.cfg.dynamics.use_macro_actions:
            self._process_macro_action(agent)
        else:
            self._process_legacy_action(agent)
        self._mark_agent_processed()

    def _begin_step_if_needed(self) -> None:
        if self._agents_processed_this_step == 0:
            if self._needs_metric_commit:
                self.previous_metrics = self._clone_metrics(self.current_metrics)
                self._needs_metric_commit = False
            self._step_requests = {}
            for name in self.kick_fired:
                self.kick_fired[name].zero_()

    def _mark_agent_processed(self) -> None:
        self._agents_processed_this_step += 1
        if self._agents_processed_this_step == len(self.attackers) + len(self.defenders):
            self._apply_ball_effects_pre_step()

    def _empty_request(self, agent: Agent) -> dict[str, Tensor]:
        zeros = torch.zeros(self.world.batch_dim, device=self.world.device)
        return {
            "kick_request": zeros.bool(),
            "dribble_request": zeros.bool(),
            "kick_alignment": zeros,
        }

    def _store_request(
        self,
        agent: Agent,
        *,
        kick_request: Tensor,
        dribble_request: Tensor,
        kick_alignment: Tensor,
    ) -> None:
        self._step_requests[agent.name] = {
            "kick_request": kick_request.bool(),
            "dribble_request": dribble_request.bool(),
            "kick_alignment": kick_alignment,
        }

    def _process_unified_action(self, agent: Agent) -> None:
        dc = self.cfg.dynamics
        fc = self.cfg.field
        u = agent.action.u

        target_pos = u[:, 0:2]
        target_oren = u[:, 2:3]
        kick_intent = u[:, 3]
        target_pos = target_pos.clamp(
            min=torch.tensor([-fc.half_length, -fc.half_width], device=u.device),
            max=torch.tensor([fc.half_length, fc.half_width], device=u.device),
        )
        nav_target, target_oren, _ = self._capture_control_targets(agent, target_pos, target_oren)

        desired_vel = self.pid_trans.calculate(agent.state.pos, nav_target, agent.name)
        desired_ang = self.pid_oren.calculate(agent.state.rot, target_oren, agent.name)
        low_level = self._to_low_level_command(agent, desired_vel, desired_ang)
        agent.action.u = torch.cat([low_level, kick_intent.unsqueeze(-1)], dim=-1)

        to_target = target_pos - agent.state.pos
        to_target = to_target / torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-8)
        facing = self._agent_facing(agent)
        kick_alignment = (facing * to_target).sum(dim=-1)
        self._store_request(
            agent,
            kick_request=kick_intent > dc.kick_intent_threshold,
            dribble_request=torch.ones_like(kick_intent, dtype=torch.bool),
            kick_alignment=kick_alignment,
        )

    def _process_macro_action(self, agent: Agent) -> None:
        dc = self.cfg.dynamics
        fc = self.cfg.field
        u = agent.action.u
        action_selector = u[:, 0]
        action_idx = ((action_selector + 1.0) * dc.n_macro_actions / 2.0).long().clamp(0, dc.n_macro_actions - 1)
        self.current_macro_action[agent.name] = action_idx

        target_pos = u[:, 1:3].clamp(
            min=torch.tensor([-fc.half_length, -fc.half_width], device=u.device),
            max=torch.tensor([fc.half_length, fc.half_width], device=u.device),
        )
        ball_pos = self.ball.state.pos
        has_ball = self._has_ball(agent)

        is_go = action_idx == MacroAction.GO_TO_BALL
        is_kick = action_idx == MacroAction.KICK_TO
        is_drib = action_idx == MacroAction.DRIBBLE_TO

        capture_request = is_go | is_kick | is_drib
        ball_nav = self._capture_nav_target(agent, ball_pos, force_capture=capture_request)
        drib_nav = torch.where(has_ball.unsqueeze(-1), target_pos, ball_nav)
        nav_target = torch.where(
            is_go.unsqueeze(-1) | is_kick.unsqueeze(-1),
            ball_nav,
            torch.where(is_drib.unsqueeze(-1), drib_nav, target_pos),
        )

        to_ball = ball_pos - agent.state.pos
        to_target = target_pos - agent.state.pos
        ball_oren = torch.atan2(to_ball[:, 1], to_ball[:, 0]).unsqueeze(-1)
        target_oren = torch.atan2(to_target[:, 1], to_target[:, 0]).unsqueeze(-1)
        drib_oren = torch.where(has_ball.unsqueeze(-1), target_oren, ball_oren)
        oren_target = torch.where(
            is_go.unsqueeze(-1),
            ball_oren,
            torch.where(is_kick.unsqueeze(-1), target_oren, torch.where(is_drib.unsqueeze(-1), drib_oren, target_oren)),
        )

        desired_vel = self.pid_trans.calculate(agent.state.pos, nav_target, agent.name)
        desired_ang = self.pid_oren.calculate(agent.state.rot, oren_target, agent.name)
        desired_vel = torch.where(
            (is_kick & has_ball).unsqueeze(-1),
            torch.zeros_like(desired_vel),
            desired_vel,
        )
        agent.action.u = self._to_low_level_command(agent, desired_vel, desired_ang)

        facing = self._agent_facing(agent)
        to_target = to_target / torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-8)
        kick_alignment = (facing * to_target).sum(dim=-1)
        self._store_request(
            agent,
            kick_request=is_kick,
            dribble_request=is_go | is_kick | is_drib,
            kick_alignment=kick_alignment,
        )

    def _process_legacy_action(self, agent: Agent) -> None:
        dc = self.cfg.dynamics
        fc = self.cfg.field
        u = agent.action.u
        target_pos = (agent.state.pos + u[:, 0:2]).clamp(
            min=torch.tensor([-fc.half_length, -fc.half_width], device=u.device),
            max=torch.tensor([fc.half_length, fc.half_width], device=u.device),
        )
        target_oren = u[:, 2:3]
        nav_target, target_oren, _ = self._capture_control_targets(agent, target_pos, target_oren)

        desired_vel = self.pid_trans.calculate(agent.state.pos, nav_target, agent.name)
        desired_ang = self.pid_oren.calculate(agent.state.rot, target_oren, agent.name)

        has_ball = self._has_ball(agent)
        kick_mode = (u[:, 3] > 0) & has_ball
        turn_mode = (u[:, 5] > 0) & has_ball & ~kick_mode
        if turn_mode.any():
            to_ball = self.ball.state.pos - agent.state.pos
            to_ball_dist = torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-6)
            to_ball_dir = to_ball / to_ball_dist
            perp_dir = torch.stack([-to_ball_dir[:, 1], to_ball_dir[:, 0]], dim=-1)
            desired_vel = torch.where(
                turn_mode.unsqueeze(-1),
                perp_dir * desired_ang * to_ball_dist * dc.turn_on_spot_radius_modifier,
                desired_vel,
            )

        low_level = self._to_low_level_command(agent, desired_vel, desired_ang)
        agent.action.u = torch.cat([low_level, u[:, 3:]], dim=-1)

        self._store_request(
            agent,
            kick_request=u[:, 3] > 0,
            dribble_request=(u[:, 4] > 0) | turn_mode,
            kick_alignment=torch.ones(self.world.batch_dim, device=self.world.device),
        )

    def _to_low_level_command(self, agent: Agent, desired_vel: Tensor, desired_ang: Tensor) -> Tensor:
        dc = self.cfg.dynamics
        fc = self.cfg.field
        if dc.physics_mode == "kinematic_legacy":
            return torch.cat([desired_vel, desired_ang], dim=-1)

        current_vel = agent.state.vel
        desired_acc = (desired_vel - current_vel) / dc.dt
        acc_norm = torch.norm(desired_acc, dim=-1, keepdim=True)
        acc_scale = torch.where(
            acc_norm > dc.robot_max_acceleration,
            dc.robot_max_acceleration / acc_norm.clamp(min=1e-8),
            torch.ones_like(acc_norm),
        )
        desired_acc = desired_acc * acc_scale
        force = desired_acc * dc.robot_mass

        current_ang = (
            agent.state.ang_vel
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None
            else torch.zeros(self.world.batch_dim, 1, device=self.world.device)
        )
        desired_ang_acc = (desired_ang - current_ang) / dc.dt
        desired_ang_acc = desired_ang_acc.clamp(
            -dc.robot_max_angular_acceleration,
            dc.robot_max_angular_acceleration,
        )
        torque = desired_ang_acc * dc.robot_mass * (fc.robot_radius**2)
        return torch.cat([force, torque], dim=-1)

    def _agent_facing(self, agent: Agent) -> Tensor:
        rot = agent.state.rot.squeeze(-1)
        return torch.stack([torch.cos(rot), torch.sin(rot)], dim=-1)

    def _apply_ball_effects_pre_step(self) -> None:
        dc = self.cfg.dynamics
        effective_holder = self.confirmed_holder

        for name in self.kick_cooldown:
            self.kick_cooldown[name] = (self.kick_cooldown[name] - 1).clamp(min=0)

        if hasattr(self.ball.state, "force") and self.ball.state.force is not None:
            self.ball.state.force = torch.zeros_like(self.ball.state.force)

        for holder_id, agent in enumerate(self.attackers + self.defenders):
            req = self._step_requests.get(agent.name, self._empty_request(agent))
            holder_mask = effective_holder == holder_id
            aligned = req["kick_alignment"] > dc.kick_align_threshold
            can_kick = holder_mask & req["kick_request"] & aligned & (self.kick_cooldown[agent.name] <= 0)
            self.kick_fired[agent.name] = can_kick
            if can_kick.any():
                kick_vel = self._agent_facing(agent) * dc.kick_speed
                self.ball.state.vel = torch.where(can_kick.unsqueeze(-1), kick_vel, self.ball.state.vel)
                self.kick_cooldown[agent.name] = torch.where(
                    can_kick,
                    torch.full_like(self.kick_cooldown[agent.name], dc.kick_cooldown_steps),
                    self.kick_cooldown[agent.name],
                )

        self._agents_processed_this_step = 0

    def _apply_ball_effects_post_step(self) -> None:
        contact_dist = self.cfg.field.robot_radius + self.cfg.field.ball_radius
        for holder_id, agent in enumerate(self.attackers + self.defenders):
            req = self._step_requests.get(agent.name, self._empty_request(agent))
            dribble_mask = (self.confirmed_holder == holder_id) & req["dribble_request"] & ~self.kick_fired[agent.name]
            if not dribble_mask.any():
                continue

            front_offset = self._agent_facing(agent) * contact_dist
            dribble_target = agent.state.pos + front_offset
            self.ball.state.pos = torch.where(dribble_mask.unsqueeze(-1), dribble_target, self.ball.state.pos)
            self.ball.state.vel = torch.where(dribble_mask.unsqueeze(-1), agent.state.vel, self.ball.state.vel)

    def _compute_candidate_holder(self) -> Tensor:
        if not self._agent_names:
            return torch.full((self.world.batch_dim,), -1, device=self.world.device, dtype=torch.long)
        contact_dist = self.cfg.field.robot_radius + self.cfg.field.ball_radius
        control_dist = min(self.cfg.dynamics.dribble_dist_threshold, contact_dist)
        cone_cos = self.cfg.dynamics.possession_cone_cos
        dists = []
        for agent in self.attackers + self.defenders:
            dist = torch.norm(self.ball.state.pos - agent.state.pos, dim=-1)
            eligible = (dist <= control_dist) & (self._facing_cos(agent, self.ball.state.pos) >= cone_cos)
            dists.append(torch.where(eligible, dist, torch.full_like(dist, float("inf"))))
        stacked = torch.stack(dists, dim=-1)
        best_dist, best_idx = stacked.min(dim=-1)
        return torch.where(best_dist.isfinite(), best_idx.long(), torch.full_like(best_idx, -1))

    def post_step(self):
        self.global_frame += 1
        self.steps += 1
        for d in self.defenders:
            self.prev_defender_vel[d.name] = d.state.vel.clone()
            self.prev_defender_rot[d.name] = d.state.rot.clone()

        ball_speed = torch.norm(self.ball.state.vel, dim=-1, keepdim=True)
        excess = ball_speed > self.cfg.dynamics.max_ball_speed
        if excess.any():
            scale = torch.where(
                excess,
                self.cfg.dynamics.max_ball_speed / ball_speed.clamp(min=1e-8),
                torch.ones_like(ball_speed),
            )
            self.ball.state.vel = self.ball.state.vel * scale

        self._apply_ball_effects_post_step()
        self._resolve_ball_penetration()
        self._update_possession_and_pass_state()
        self.current_metrics = self._compute_metrics()
        self._needs_metric_commit = True

    def _update_possession_and_pass_state(self) -> None:
        dc = self.cfg.dynamics
        candidate = self._compute_candidate_holder()
        switched = candidate != self.candidate_holder
        self.candidate_holder = torch.where(switched, candidate, self.candidate_holder)
        self.candidate_frames = torch.where(
            switched,
            torch.ones_like(self.candidate_frames),
            self.candidate_frames + 1,
        )

        self.holder_switch_cooldown = (self.holder_switch_cooldown - 1).clamp(min=0)
        ready_to_switch = (
            (candidate != self.confirmed_holder)
            & (self.candidate_frames >= dc.pass_confirm_frames)
            & (self.holder_switch_cooldown <= 0)
        )

        previous_holder = self.confirmed_holder.clone()
        new_holder = torch.where(ready_to_switch, candidate, self.confirmed_holder)
        released_from_passer = ready_to_switch & (previous_holder == 0) & (new_holder != 0)
        if released_from_passer.any():
            self.pass_active = torch.where(released_from_passer, torch.ones_like(self.pass_active), self.pass_active)
            self.pass_start_ball_pos = torch.where(
                released_from_passer.unsqueeze(-1),
                self.ball.state.pos,
                self.pass_start_ball_pos,
            )

        travel = torch.norm(self.ball.state.pos - self.pass_start_ball_pos, dim=-1)
        completed = ready_to_switch & (new_holder == 1) & self.pass_active & (travel >= dc.pass_min_ball_travel)
        self.pass_completed = self.pass_completed | completed
        self.pass_count = self.pass_count + completed.long()

        defender_pickup = ready_to_switch & (new_holder >= self.cfg.n_attackers)
        self.ball_intercepted = self.ball_intercepted | defender_pickup

        attacker_switch = ready_to_switch & (new_holder >= 0) & (new_holder < self.cfg.n_attackers)
        self.last_attacker_holder = torch.where(attacker_switch, new_holder, self.last_attacker_holder)
        self.pass_active = torch.where(
            completed | defender_pickup | ((ready_to_switch) & (new_holder == 0)),
            torch.zeros_like(self.pass_active),
            self.pass_active,
        )
        self.confirmed_holder = new_holder
        self.holder_switch_cooldown = torch.where(
            ready_to_switch,
            torch.full_like(self.holder_switch_cooldown, dc.holder_switch_cooldown_steps),
            self.holder_switch_cooldown,
        )

    def observation(self, agent: Agent) -> Tensor:
        fc = self.cfg.field
        dc = self.cfg.dynamics
        pos_scale = torch.tensor([fc.half_length, fc.half_width], device=self.world.device)
        vel_scale = max(dc.robot_max_speed, 1e-6)
        ball_vel_scale = max(dc.max_ball_speed, 1e-6)

        obs_parts: list[Tensor] = []
        obs_parts.append(agent.state.pos / pos_scale)
        obs_parts.append(agent.state.vel / vel_scale)
        ang_vel = (
            agent.state.ang_vel
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None
            else torch.zeros(self.world.batch_dim, 1, device=self.world.device)
        )
        obs_parts.append(ang_vel / max(dc.robot_max_angular_vel, 1e-6))
        rot = agent.state.rot.squeeze(-1)
        obs_parts.append(torch.stack([torch.cos(rot), torch.sin(rot)], dim=-1))

        ball_rel = (self.ball.state.pos - agent.state.pos) / pos_scale
        obs_parts.append(ball_rel)
        obs_parts.append(self.ball.state.vel / ball_vel_scale)
        obs_parts.append(torch.norm(self.ball.state.vel, dim=-1, keepdim=True) / ball_vel_scale)

        holder_id = self._holder_name_to_id.get(agent.name, -1)
        obs_parts.append(self._has_ball(agent).unsqueeze(-1).float())
        obs_parts.append((self.confirmed_holder == holder_id).unsqueeze(-1).float())
        obs_parts.append(self._facing_cos(agent, self.ball.state.pos).unsqueeze(-1))

        az_cx = fc.active_zone_center_x
        az_cy = fc.active_zone_center_y
        pos = agent.state.pos
        dist_right = ((az_cx + fc.active_zone_half_length) - pos[:, 0:1]) / fc.active_zone_half_length
        dist_left = (pos[:, 0:1] - (az_cx - fc.active_zone_half_length)) / fc.active_zone_half_length
        dist_top = ((az_cy + fc.active_zone_half_width) - pos[:, 1:2]) / fc.active_zone_half_width
        dist_bottom = (pos[:, 1:2] - (az_cy - fc.active_zone_half_width)) / fc.active_zone_half_width
        obs_parts.extend([dist_right, dist_left, dist_top, dist_bottom])

        is_attacker = agent in self.attackers
        teammates = self.attackers if is_attacker else self.defenders
        opponents = self.defenders if is_attacker else self.attackers

        for t in teammates:
            if t is agent:
                continue
            obs_parts.append((t.state.pos - agent.state.pos) / pos_scale)
            obs_parts.append(t.state.vel / vel_scale)

        for o in opponents:
            obs_parts.append((o.state.pos - agent.state.pos) / pos_scale)
            obs_parts.append(o.state.vel / vel_scale)
            if is_attacker and o.name in self.prev_defender_vel:
                obs_parts.append(self.prev_defender_vel[o.name] / vel_scale)
                obs_parts.append(self.prev_defender_rot[o.name] / math.pi)

        time_remaining = (1.0 - self.steps.float() / self.cfg.max_steps).unsqueeze(-1)
        obs_parts.append(time_remaining)
        return torch.cat(obs_parts, dim=-1)

    def reward(self, agent: Agent) -> Tensor:
        reward, components = compute_passing_reward(
            agent_name=agent.name,
            agent_pos=agent.state.pos,
            ball_pos=self.ball.state.pos,
            scenario=self,
        )
        self._reward_components[agent.name] = components
        return reward

    def done(self) -> Tensor:
        fc = self.cfg.field
        ball_pos = self.ball.state.pos
        oob = (ball_pos[:, 0].abs() > fc.half_length + fc.boundary_margin) | (
            ball_pos[:, 1].abs() > fc.half_width + fc.boundary_margin
        )
        timeout = self.steps >= self.cfg.max_steps
        return self.pass_completed | self.ball_intercepted | oob | timeout

    def info(self, agent: Agent) -> dict:
        info = {
            "pass_completed": self.pass_completed,
            "ball_intercepted": self.ball_intercepted,
            "pass_count": self.pass_count,
            "steps": self.steps,
            "confirmed_holder": self.confirmed_holder,
        }
        info.update(self._reward_components.get(agent.name, {}))
        return info
