"""VMAS scenario for ASPAC SSL passing drills.

Action space: 5D [vx, vy, omega (continuous), kick (binary impulse), dribble (binary on/off)]
- Dims 0-2: consumed by VelocityHolonomic dynamics (sets agent vel directly)
- Dim 3: kick — thresholded > 0, applies one-shot impulse to ball
- Dim 4: dribble — thresholded > 0, attracts ball toward agent front

Supports 2v0, 2v1, 2v2 configurations with ASPAC reward hooks:
- Delta-based dense shaping (standing still = 0 reward)
- Stackelberg timing (attackers see defender's t-1 velocity/rotation)
- Episodic Envy-Free bonus at done()
"""

import math

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario

from utama_core.training.scenario.passing_config import PassingScenarioConfig
from utama_core.training.scenario.passing_rewards import (
    compute_passing_reward,
    envy_free_bonus,
)
from utama_core.training.scenario.velocity_holonomic import VelocityHolonomic


class PassingScenario(BaseScenario):
    """ASPAC passing drill scenario.

    Reward structure:
    - Dense: delta-based ball-to-receiver / receiver-to-ball shaping
    - Sparse: successful pass (+50), ball OOB (-5)
    - ASPAC stubs: envy-free bonus (episodic), deception penalty
    """

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

        # --- Attacker agents ---
        self.attackers: list[Agent] = []
        for i in range(self.cfg.n_attackers):
            agent = Agent(
                name=f"attacker_{i}",
                shape=Sphere(radius=fc.robot_radius),
                movable=True,
                rotatable=True,
                collide=True,
                mass=dc.robot_mass,
                max_speed=None,
                color=torch.tensor([0.2, 0.2, 0.9]),
                dynamics=VelocityHolonomic(
                    max_speed=dc.robot_max_speed,
                    max_angular_vel=dc.robot_max_angular_vel,
                ),
                action_size=5,
                drag=0,
            )
            world.add_agent(agent)
            self.attackers.append(agent)

        # --- Defender agents ---
        self.defenders: list[Agent] = []
        for i in range(self.cfg.n_defenders):
            agent = Agent(
                name=f"defender_{i}",
                shape=Sphere(radius=fc.robot_radius),
                movable=True,
                rotatable=True,
                collide=True,
                mass=dc.robot_mass,
                max_speed=None,
                color=torch.tensor([0.9, 0.9, 0.2]),
                dynamics=VelocityHolonomic(
                    max_speed=dc.robot_max_speed,
                    max_angular_vel=dc.robot_max_angular_vel,
                ),
                action_size=5,
                drag=0,
            )
            world.add_agent(agent)
            self.defenders.append(agent)

        # --- Ball ---
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

        # --- Boundary walls ---
        self._create_walls(world, fc)

        # --- State tracking ---
        self.steps = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.pass_completed = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.ball_intercepted = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.pass_count = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.last_holder = torch.full((batch_dim,), -1, device=device, dtype=torch.long)

        # Delta-based shaping: previous distances
        self.prev_ball_to_receiver_dist = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.prev_receiver_to_ball_dist = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.prev_passer_to_ball_dist = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        # Cumulative rewards for episodic Envy-Free
        self.cumulative_attacker_rewards: dict[str, torch.Tensor] = {
            a.name: torch.zeros(batch_dim, device=device) for a in self.attackers
        }

        # Stackelberg timing: previous-step defender state
        self.prev_defender_vel: dict[str, torch.Tensor] = {
            d.name: torch.zeros(batch_dim, 2, device=device) for d in self.defenders
        }
        self.prev_defender_rot: dict[str, torch.Tensor] = {
            d.name: torch.zeros(batch_dim, 1, device=device) for d in self.defenders
        }

        return world

    # ------------------------------------------------------------------
    # Field construction
    # ------------------------------------------------------------------

    def _create_walls(self, world: World, fc):
        """Create boundary walls around the field."""
        wall_thickness = 0.05
        self.walls: list[tuple] = []

        for name, y_sign in [("top", 1), ("bottom", -1)]:
            wall = Landmark(
                name=f"wall_{name}",
                shape=Box(
                    length=2 * (fc.half_length + fc.boundary_margin),
                    width=wall_thickness,
                ),
                movable=False,
                collide=True,
                color=torch.tensor([0.3, 0.6, 0.3]),
            )
            world.add_landmark(wall)
            self.walls.append((name, wall, 0.0, y_sign * fc.half_width))

        for name, x_sign in [("left", -1), ("right", 1)]:
            wall = Landmark(
                name=f"wall_{name}",
                shape=Box(
                    length=wall_thickness,
                    width=2 * (fc.half_width + fc.boundary_margin),
                ),
                movable=False,
                collide=True,
                color=torch.tensor([0.3, 0.6, 0.3]),
            )
            world.add_landmark(wall)
            self.walls.append((name, wall, x_sign * fc.half_length, 0.0))

    # ------------------------------------------------------------------
    # Reset — ASPAC geometric initialization
    # ------------------------------------------------------------------

    def reset_world_at(self, env_index: int = None):
        fc = self.cfg.field
        device = self.world.device

        # Fixed strategic positions
        ball_pos = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32)
        goal_center = torch.tensor([fc.half_length, 0.0], device=device, dtype=torch.float32)

        # --- Ball ---
        self._set_entity_state(self.ball, ball_pos[0].item(), ball_pos[1].item(), env_index)

        # --- Attacker 0 (passer): 0.3m behind ball with angular randomization ---
        if self.cfg.n_attackers >= 1:
            a0 = self.attackers[0]
            if env_index is None:
                beta = torch.empty(self.world.batch_dim, 1, device=device).uniform_(0, math.pi / 4)
            else:
                beta = torch.empty(1, device=device).uniform_(0, math.pi / 4)

            offset_x = 0.3 * torch.cos(beta).squeeze(-1)
            offset_y = 0.3 * torch.sin(beta).squeeze(-1)

            if env_index is None:
                a0_x = ball_pos[0] - offset_x
                a0_y = ball_pos[1] - offset_y
                a0_pos = torch.stack([a0_x, a0_y], dim=-1)
                a0.set_pos(a0_pos, batch_index=None)
                a0.set_vel(
                    torch.zeros(self.world.batch_dim, 2, device=device),
                    batch_index=None,
                )
                dir_to_goal = goal_center.unsqueeze(0) - a0_pos
                rot = torch.atan2(dir_to_goal[:, 1], dir_to_goal[:, 0]).unsqueeze(-1)
                a0.set_rot(rot, batch_index=None)
            else:
                a0_x = ball_pos[0] - offset_x
                a0_y = ball_pos[1] - offset_y
                a0_pos_s = torch.stack([a0_x, a0_y])
                a0.set_pos(a0_pos_s, batch_index=env_index)
                a0.set_vel(torch.zeros(2, device=device), batch_index=env_index)
                dir_to_goal = goal_center - a0_pos_s
                rot = torch.atan2(dir_to_goal[1], dir_to_goal[0]).unsqueeze(0)
                a0.set_rot(rot, batch_index=env_index)

        # --- Attacker 1 (receiver): fixed y=-2.5, randomized x ∈ [1.0, 2.0] ---
        if self.cfg.n_attackers >= 2:
            a1 = self.attackers[1]
            if env_index is None:
                a1_x = torch.empty(self.world.batch_dim, device=device).uniform_(1.0, 2.0)
                a1_y = torch.full((self.world.batch_dim,), -2.5, device=device)
                a1_pos = torch.stack([a1_x, a1_y], dim=-1)
                a1.set_pos(a1_pos, batch_index=None)
                a1.set_vel(
                    torch.zeros(self.world.batch_dim, 2, device=device),
                    batch_index=None,
                )
                a1.set_rot(
                    torch.zeros(self.world.batch_dim, 1, device=device),
                    batch_index=None,
                )
            else:
                a1_x = torch.empty(1, device=device).uniform_(1.0, 2.0)
                a1_pos_s = torch.cat([a1_x, torch.tensor([-2.5], device=device)])
                a1.set_pos(a1_pos_s, batch_index=env_index)
                a1.set_vel(torch.zeros(2, device=device), batch_index=env_index)
                a1.set_rot(torch.zeros(1, device=device), batch_index=env_index)

        # --- Defender 0: 0.75m from ball on vector to goal center ---
        if self.cfg.n_defenders >= 1:
            d0 = self.defenders[0]
            vec_to_goal = goal_center - ball_pos
            norm_vec = vec_to_goal / torch.norm(vec_to_goal)
            d0_pos_base = ball_pos + norm_vec * 0.75

            if env_index is None:
                d0_pos = d0_pos_base.unsqueeze(0).expand(self.world.batch_dim, -1)
                d0.set_pos(d0_pos, batch_index=None)
                d0.set_vel(
                    torch.zeros(self.world.batch_dim, 2, device=device),
                    batch_index=None,
                )
                d0.set_rot(
                    torch.zeros(self.world.batch_dim, 1, device=device),
                    batch_index=None,
                )
            else:
                d0.set_pos(d0_pos_base, batch_index=env_index)
                d0.set_vel(torch.zeros(2, device=device), batch_index=env_index)
                d0.set_rot(torch.zeros(1, device=device), batch_index=env_index)

        # --- Defender 1: fixed y=-2.0, randomized x ∈ [1.0, 2.5] ---
        if self.cfg.n_defenders >= 2:
            d1 = self.defenders[1]
            if env_index is None:
                d1_x = torch.empty(self.world.batch_dim, device=device).uniform_(1.0, 2.5)
                d1_y = torch.full((self.world.batch_dim,), -2.0, device=device)
                d1_pos = torch.stack([d1_x, d1_y], dim=-1)
                d1.set_pos(d1_pos, batch_index=None)
                d1.set_vel(
                    torch.zeros(self.world.batch_dim, 2, device=device),
                    batch_index=None,
                )
                d1.set_rot(
                    torch.zeros(self.world.batch_dim, 1, device=device),
                    batch_index=None,
                )
            else:
                d1_x = torch.empty(1, device=device).uniform_(1.0, 2.5)
                d1_pos_s = torch.cat([d1_x, torch.tensor([-2.0], device=device)])
                d1.set_pos(d1_pos_s, batch_index=env_index)
                d1.set_vel(torch.zeros(2, device=device), batch_index=env_index)
                d1.set_rot(torch.zeros(1, device=device), batch_index=env_index)

        # --- Position walls ---
        for _, wall, wx, wy in self.walls:
            self._set_entity_state(wall, wx, wy, env_index, movable=False)

        # --- Reset tracking ---
        if env_index is None:
            self.pass_completed.zero_()
            self.ball_intercepted.zero_()
            self.pass_count.zero_()
            self.last_holder.fill_(-1)
            # Stagger initial steps for smooth logging
            self.steps = torch.randint(
                0,
                self.cfg.max_steps,
                (self.world.batch_dim,),
                device=device,
                dtype=torch.long,
            )
            # Initialize previous distances for delta-based shaping
            if self.cfg.n_attackers >= 1:
                passer_pos = self.attackers[0].state.pos
                bp = self.ball.state.pos
                self.prev_passer_to_ball_dist = torch.norm(passer_pos - bp, dim=-1)
            if self.cfg.n_attackers >= 2:
                receiver_pos = self.attackers[1].state.pos
                bp = self.ball.state.pos
                self.prev_ball_to_receiver_dist = torch.norm(bp - receiver_pos, dim=-1)
                self.prev_receiver_to_ball_dist = torch.norm(receiver_pos - bp, dim=-1)
            for name in self.cumulative_attacker_rewards:
                self.cumulative_attacker_rewards[name].zero_()
            for name in self.prev_defender_vel:
                self.prev_defender_vel[name].zero_()
            for name in self.prev_defender_rot:
                self.prev_defender_rot[name].zero_()
        else:
            self.pass_completed[env_index] = False
            self.ball_intercepted[env_index] = False
            self.pass_count[env_index] = 0
            self.last_holder[env_index] = -1
            self.steps[env_index] = torch.randint(0, self.cfg.max_steps, (1,), device=device).item()
            if self.cfg.n_attackers >= 1:
                passer_pos = self.attackers[0].state.pos[env_index]
                bp = self.ball.state.pos[env_index]
                self.prev_passer_to_ball_dist[env_index] = torch.norm(passer_pos - bp)
            if self.cfg.n_attackers >= 2:
                receiver_pos = self.attackers[1].state.pos[env_index]
                bp = self.ball.state.pos[env_index]
                self.prev_ball_to_receiver_dist[env_index] = torch.norm(bp - receiver_pos)
                self.prev_receiver_to_ball_dist[env_index] = torch.norm(receiver_pos - bp)
            for name in self.cumulative_attacker_rewards:
                self.cumulative_attacker_rewards[name][env_index] = 0.0
            for name in self.prev_defender_vel:
                self.prev_defender_vel[name][env_index] = 0.0
            for name in self.prev_defender_rot:
                self.prev_defender_rot[name][env_index] = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_entity_state(self, entity, x, y, env_index, movable=True):
        pos = torch.tensor([x, y], device=self.world.device, dtype=torch.float32)
        if env_index is None:
            entity.set_pos(pos.unsqueeze(0).expand(self.world.batch_dim, -1), batch_index=None)
            if movable:
                zero2 = torch.zeros(self.world.batch_dim, 2, device=self.world.device, dtype=torch.float32)
                entity.set_vel(zero2, batch_index=None)
        else:
            entity.set_pos(pos, batch_index=env_index)
            if movable:
                zero2 = torch.zeros(2, device=self.world.device, dtype=torch.float32)
                entity.set_vel(zero2, batch_index=env_index)

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def process_action(self, agent: Agent):
        """Handle kick/dribble (dims 3-4). Zero out fixed defender actions."""
        # Fixed defenders: zero all actions
        if agent in self.defenders and self.cfg.defender_behavior == "fixed":
            agent.action.u = torch.zeros_like(agent.action.u)
            return

        dc = self.cfg.dynamics
        kick_action = agent.action.u[:, 3]
        dribble_action = agent.action.u[:, 4]

        agent_pos = agent.state.pos
        ball_pos = self.ball.state.pos

        dist_to_ball = torch.norm(agent_pos - ball_pos, dim=-1)
        in_range = dist_to_ball < dc.dribble_dist_threshold

        # Kick: set ball velocity directly (matches rsim's kick_v_x command)
        kick_triggered = (kick_action > 0) & in_range
        if kick_triggered.any():
            cos_rot = torch.cos(agent.state.rot.squeeze(-1))
            sin_rot = torch.sin(agent.state.rot.squeeze(-1))
            kick_dir = torch.stack([cos_rot, sin_rot], dim=-1)
            kick_vel = kick_dir * dc.kick_speed
            self.ball.state.vel = torch.where(
                kick_triggered.unsqueeze(-1),
                kick_vel,
                self.ball.state.vel,
            )

        # Dribble: binary attract
        dribble_active = (dribble_action > 0) & in_range
        if dribble_active.any():
            fc = self.cfg.field
            cos_rot = torch.cos(agent.state.rot.squeeze(-1))
            sin_rot = torch.sin(agent.state.rot.squeeze(-1))
            front_offset = torch.stack([cos_rot, sin_rot], dim=-1) * (fc.robot_radius + fc.ball_radius)
            target_pos = agent_pos + front_offset
            attract_dir = target_pos - ball_pos
            attract_force = attract_dir * dc.dribble_force
            self.ball.state.vel = (
                self.ball.state.vel + attract_force * dribble_active.unsqueeze(-1).float() * self.world.dt
            )

    def post_step(self):
        """Update Stackelberg timing state and clamp ball speed."""
        # Update previous defender state for Stackelberg timing
        for d in self.defenders:
            self.prev_defender_vel[d.name] = d.state.vel.clone()
            self.prev_defender_rot[d.name] = d.state.rot.clone()

        # Cap ball speed (prevents unrealistic collision-launched velocities)
        dc = self.cfg.dynamics
        ball_speed = torch.norm(self.ball.state.vel, dim=-1, keepdim=True)
        excess = ball_speed > dc.max_ball_speed
        if excess.any():
            scale = torch.where(
                excess,
                dc.max_ball_speed / ball_speed.clamp(min=1e-8),
                torch.ones_like(ball_speed),
            )
            self.ball.state.vel = self.ball.state.vel * scale

        # NOTE: prev distances for delta-based reward shaping are updated in
        # reward() AFTER the reward is computed. VMAS calls post_step() before
        # reward(), so updating here would zero out all deltas.

    # ------------------------------------------------------------------
    # Observation — ego-centric with Stackelberg timing
    # ------------------------------------------------------------------

    def observation(self, agent: Agent) -> torch.Tensor:
        """Ego-centric observation with Stackelberg timing for attackers.

        Common: own vel(2) + ang_vel(1) + rot(1) + ball_rel(2) + ball_vel(2)
                + active zone edge distances(4) = 12
        Per teammate: rel_pos(2) + vel(2) = 4
        Attacker per opponent: rel_pos(2) + vel(2) + prev_vel(2) + prev_rot(1) = 7
        Defender per opponent: rel_pos(2) + vel(2) = 4
        """
        fc = self.cfg.field
        obs_parts: list[torch.Tensor] = []

        # Own state
        obs_parts.append(agent.state.vel)
        if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
            obs_parts.append(agent.state.ang_vel)
        else:
            obs_parts.append(torch.zeros(self.world.batch_dim, 1, device=self.world.device))
        obs_parts.append(agent.state.rot)

        # Ball
        ball_rel = self.ball.state.pos - agent.state.pos
        obs_parts.append(ball_rel)
        obs_parts.append(self.ball.state.vel)

        # Active zone edge distances
        az_cx = fc.active_zone_center_x
        az_cy = fc.active_zone_center_y
        pos = agent.state.pos
        dist_right = (az_cx + fc.active_zone_half_length) - pos[:, 0:1]
        dist_left = pos[:, 0:1] - (az_cx - fc.active_zone_half_length)
        dist_top = (az_cy + fc.active_zone_half_width) - pos[:, 1:2]
        dist_bottom = pos[:, 1:2] - (az_cy - fc.active_zone_half_width)
        obs_parts.extend([dist_right, dist_left, dist_top, dist_bottom])

        # Teammates
        is_attacker = agent in self.attackers
        teammates = self.attackers if is_attacker else self.defenders
        opponents = self.defenders if is_attacker else self.attackers

        for t in teammates:
            if t is agent:
                continue
            obs_parts.append(t.state.pos - agent.state.pos)
            obs_parts.append(t.state.vel)

        # Opponents — attackers get Stackelberg timing (t-1 state)
        for o in opponents:
            obs_parts.append(o.state.pos - agent.state.pos)
            obs_parts.append(o.state.vel)
            if is_attacker and o.name in self.prev_defender_vel:
                obs_parts.append(self.prev_defender_vel[o.name])
                obs_parts.append(self.prev_defender_rot[o.name])

        return torch.cat(obs_parts, dim=-1)

    # ------------------------------------------------------------------
    # Reward — delegates to passing_rewards.py
    # ------------------------------------------------------------------

    def reward(self, agent: Agent) -> torch.Tensor:
        self._update_pass_tracking()

        r = compute_passing_reward(
            agent_name=agent.name,
            agent_pos=agent.state.pos,
            ball_pos=self.ball.state.pos,
            scenario=self,
        )

        # Update prev distances AFTER computing reward so that the next step's
        # delta = (this step's distance) - (next step's distance) is non-zero.
        # VMAS calls post_step() before reward(), so we must update here.
        self._update_prev_distances(agent)

        # Accumulate for episodic Envy-Free
        if agent.name in self.cumulative_attacker_rewards:
            self.cumulative_attacker_rewards[agent.name] = self.cumulative_attacker_rewards[agent.name] + r

        return r

    def _update_prev_distances(self, agent: Agent):
        """Update previous distances for delta-based shaping after reward."""
        bp = self.ball.state.pos
        if self.cfg.n_attackers >= 1 and agent is self.attackers[0]:
            self.prev_passer_to_ball_dist = torch.norm(agent.state.pos - bp, dim=-1)
            if self.cfg.n_attackers >= 2:
                self.prev_ball_to_receiver_dist = torch.norm(bp - self.attackers[1].state.pos, dim=-1)
        if self.cfg.n_attackers >= 2 and agent is self.attackers[1]:
            self.prev_receiver_to_ball_dist = torch.norm(agent.state.pos - bp, dim=-1)

    # ------------------------------------------------------------------
    # Pass & interception tracking
    # ------------------------------------------------------------------

    def _update_pass_tracking(self):
        dc = self.cfg.dynamics
        ball_pos = self.ball.state.pos

        for i, agent in enumerate(self.attackers):
            dist = torch.norm(agent.state.pos - ball_pos, dim=-1)
            has_ball = dist < dc.dribble_dist_threshold
            new_pass = has_ball & (self.last_holder != i) & (self.last_holder >= 0)
            self.pass_completed = self.pass_completed | new_pass
            self.pass_count = self.pass_count + new_pass.long()
            self.last_holder = torch.where(has_ball, i, self.last_holder)

        # Interception: any defender within dribble range of ball
        for d in self.defenders:
            dist_to_ball = torch.norm(ball_pos - d.state.pos, dim=-1)
            newly_intercepted = ~self.ball_intercepted & (dist_to_ball < dc.dribble_dist_threshold)
            self.ball_intercepted = self.ball_intercepted | newly_intercepted

    # ------------------------------------------------------------------
    # Done — with episodic Envy-Free bonus
    # ------------------------------------------------------------------

    def done(self) -> torch.Tensor:
        self.steps += 1
        fc = self.cfg.field
        ball_pos = self.ball.state.pos

        oob = (ball_pos[:, 0].abs() > fc.half_length + fc.boundary_margin) | (
            ball_pos[:, 1].abs() > fc.half_width + fc.boundary_margin
        )
        timeout = self.steps >= self.cfg.max_steps
        is_done = self.pass_completed | self.ball_intercepted | oob | timeout

        # Apply episodic Envy-Free bonus at termination
        if self.cfg.rewards.envy_free_weight > 0 and is_done.any():
            envy_free_bonus(
                self.cumulative_attacker_rewards,
                self.cfg.rewards.envy_free_weight,
            )

        return is_done

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self, agent: Agent) -> dict:
        return {
            "pass_completed": self.pass_completed,
            "ball_intercepted": self.ball_intercepted,
            "pass_count": self.pass_count,
            "steps": self.steps,
        }
