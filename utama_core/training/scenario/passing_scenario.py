"""VMAS scenario for ASPAC SSL passing drills.

Supports three action space modes (select via config flags):

**Unified mode (4D, default):** [target_x, target_y, target_oren, kick_intent]
  - target_x, target_y are absolute field positions (navigation target)
  - target_oren is desired facing direction in [-pi, pi]
  - kick_intent > 0 triggers kick when aligned + has ball (2-frame dribbler release)
  - Auto-dribble when ball is in front area of robot
  - PID gains from ``get_pid_configs(Mode.VMAS)`` via BatchedPID

**Macro-action mode (3D):** [action_selector, target_x, target_y]
  - action_selector in [-1, 1] mapped to 4 discrete macro-actions via bins
  - Legacy: kept for backward compatibility / A-B comparison

**Legacy mode (6D):** [delta_x, delta_y, target_oren, kick, dribble, turn_on_spot]
  - Legacy: kept for backward compatibility

Supports 2v0, 2v1, 2v2 configurations with ASPAC reward hooks:
- Delta-based dense shaping (standing still = 0 reward)
- Stackelberg timing (attackers see defender's t-1 velocity/rotation)
- Episodic Envy-Free bonus at done()
"""

import math

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario

from utama_core.training.scenario.batched_pid import (
    BatchedOrientationPID,
    BatchedTranslationPID,
)
from utama_core.training.scenario.passing_config import (
    MacroAction,
    PassingScenarioConfig,
)
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

        # --- Action space configuration ---
        if dc.use_unified_actions:
            _action_size = 4
            _u_multiplier = [
                fc.half_length,  # target_x: absolute field position
                fc.half_width,  # target_y: absolute field position
                math.pi,  # target_oren: [-pi, pi] rad
                1.0,  # kick_intent: [-1, 1]
            ]
        elif dc.use_macro_actions:
            _action_size = 3
            _u_multiplier = [
                1.0,  # action_selector: [-1, 1] → macro-action bins
                fc.half_length,  # target_x: absolute field position
                fc.half_width,  # target_y: absolute field position
            ]
        else:
            _action_size = 6
            _u_multiplier = [
                dc.action_delta_range,  # delta_x (meters)
                dc.action_delta_range,  # delta_y (meters)
                math.pi,  # target_oren (radians)
                1.0,  # kick trigger
                1.0,  # dribble trigger
                1.0,  # turn_on_spot trigger
            ]

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
                action_size=_action_size,
                u_multiplier=_u_multiplier,
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
                action_size=_action_size,
                u_multiplier=_u_multiplier,
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

        # Delta-based shaping: previous distances and facing
        self.prev_ball_to_receiver_dist = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.prev_receiver_to_ball_dist = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.prev_passer_to_ball_dist = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.prev_receiver_facing_ball_cos = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.prev_passer_facing_receiver_cos = torch.zeros(batch_dim, device=device, dtype=torch.float32)

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

        # PID controllers (shared across all action modes)
        all_agents = self.attackers + self.defenders
        self.pid_trans = BatchedTranslationPID()
        self.pid_oren = BatchedOrientationPID()

        # Unified action space state: 2-frame dribbler-release kick
        if dc.use_unified_actions:
            self.kick_pending: dict[str, torch.Tensor] = {
                a.name: torch.zeros(batch_dim, device=device, dtype=torch.bool) for a in all_agents
            }

        # Macro-action state tracking (for logging/debugging)
        if dc.use_macro_actions:
            self.current_macro_action: dict[str, torch.Tensor] = {
                a.name: torch.zeros(batch_dim, device=device, dtype=torch.long) for a in all_agents
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
            self.steps.zero_()
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
                self.prev_receiver_facing_ball_cos = self._compute_facing_ball_cos(self.attackers[1], bp)
                self.prev_passer_facing_receiver_cos = self._compute_facing_ball_cos(self.attackers[0], receiver_pos)
            for name in self.cumulative_attacker_rewards:
                self.cumulative_attacker_rewards[name].zero_()
            for name in self.prev_defender_vel:
                self.prev_defender_vel[name].zero_()
            for name in self.prev_defender_rot:
                self.prev_defender_rot[name].zero_()
            # Reset PID controllers
            for a in self.attackers + self.defenders:
                self.pid_trans.reset(a.name)
                self.pid_oren.reset(a.name)
            if hasattr(self, "kick_pending"):
                for name in self.kick_pending:
                    self.kick_pending[name].zero_()
            if hasattr(self, "current_macro_action"):
                for name in self.current_macro_action:
                    self.current_macro_action[name].zero_()
        else:
            self.pass_completed[env_index] = False
            self.ball_intercepted[env_index] = False
            self.pass_count[env_index] = 0
            self.last_holder[env_index] = -1
            self.steps[env_index] = 0
            if self.cfg.n_attackers >= 1:
                passer_pos = self.attackers[0].state.pos[env_index]
                bp = self.ball.state.pos[env_index]
                self.prev_passer_to_ball_dist[env_index] = torch.norm(passer_pos - bp)
            if self.cfg.n_attackers >= 2:
                receiver_pos = self.attackers[1].state.pos[env_index]
                bp = self.ball.state.pos[env_index]
                self.prev_ball_to_receiver_dist[env_index] = torch.norm(bp - receiver_pos)
                self.prev_receiver_to_ball_dist[env_index] = torch.norm(receiver_pos - bp)
                self.prev_receiver_facing_ball_cos[env_index] = self._compute_facing_ball_cos_single(
                    self.attackers[1], bp, env_index
                )
                self.prev_passer_facing_receiver_cos[env_index] = self._compute_facing_ball_cos_single(
                    self.attackers[0], receiver_pos, env_index
                )
            for name in self.cumulative_attacker_rewards:
                self.cumulative_attacker_rewards[name][env_index] = 0.0
            for name in self.prev_defender_vel:
                self.prev_defender_vel[name][env_index] = 0.0
            for name in self.prev_defender_rot:
                self.prev_defender_rot[name][env_index] = 0.0
            # Reset PID controllers for this env
            for a in self.attackers + self.defenders:
                self.pid_trans.reset(a.name, batch_index=env_index)
                self.pid_oren.reset(a.name, batch_index=env_index)
            if hasattr(self, "kick_pending"):
                for name in self.kick_pending:
                    self.kick_pending[name][env_index] = False
            if hasattr(self, "current_macro_action"):
                for name in self.current_macro_action:
                    self.current_macro_action[name][env_index] = 0

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
    # Helpers
    # ------------------------------------------------------------------

    def _has_ball(self, agent: Agent) -> torch.Tensor:
        """Check if the ball is within dribble range AND in front of the agent."""
        to_ball = self.ball.state.pos - agent.state.pos
        dist = torch.norm(to_ball, dim=-1)
        in_range = dist < self.cfg.dynamics.dribble_dist_threshold
        cos_rot = torch.cos(agent.state.rot.squeeze(-1))
        sin_rot = torch.sin(agent.state.rot.squeeze(-1))
        facing = torch.stack([cos_rot, sin_rot], dim=-1)
        ball_in_front = (to_ball * facing).sum(dim=-1) > 0
        return in_range & ball_in_front

    def _compute_facing_ball_cos(self, agent: Agent, ball_pos: torch.Tensor) -> torch.Tensor:
        """Cosine of angle between agent facing direction and direction to ball (batched)."""
        to_ball = ball_pos - agent.state.pos
        to_ball_norm = torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-8)
        to_ball_dir = to_ball / to_ball_norm
        rot = agent.state.rot.squeeze(-1)
        facing = torch.stack([torch.cos(rot), torch.sin(rot)], dim=-1)
        return (facing * to_ball_dir).sum(dim=-1)

    def _compute_facing_ball_cos_single(self, agent: Agent, ball_pos: torch.Tensor, env_index: int) -> torch.Tensor:
        """Cosine of angle between agent facing and ball direction for a single env."""
        to_ball = ball_pos - agent.state.pos[env_index]
        to_ball_dir = to_ball / torch.norm(to_ball).clamp(min=1e-8)
        rot = agent.state.rot[env_index].squeeze(-1)
        facing = torch.stack([torch.cos(rot), torch.sin(rot)])
        return (facing * to_ball_dir).sum()

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _angle_wrap(self, angle: torch.Tensor) -> torch.Tensor:
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def process_action(self, agent: Agent):
        """Dispatch to unified, macro-action, or legacy action processing."""
        # Fixed defenders: zero all actions
        if agent in self.defenders and self.cfg.defender_behavior == "fixed":
            agent.action.u = torch.zeros_like(agent.action.u)
            return

        dc = self.cfg.dynamics
        if dc.use_unified_actions:
            self._process_unified_action(agent)
        elif dc.use_macro_actions:
            self._process_macro_action(agent)
        else:
            self._process_legacy_action(agent)

    # ------------------------------------------------------------------
    # Unified action processing (4D: [target_x, target_y, target_oren, kick_intent])
    # ------------------------------------------------------------------

    def _process_unified_action(self, agent: Agent):
        """Convert 4D [target_x, target_y, target_oren, kick_intent] to velocity commands.

        Navigation tracks the target position.  Orientation tracks the
        explicit target_oren from the action.  Dribble auto-engages when
        the ball is in the front area of the robot.  Kick uses a 2-frame
        dribbler-release mechanic: frame 1 disengages the dribbler, frame 2
        fires the kick along the robot's facing direction.
        """
        dc = self.cfg.dynamics
        fc = self.cfg.field
        u = agent.action.u  # (batch, 4) — already scaled by u_multiplier

        # --- Parse actions ---
        target_pos = u[:, 0:2]  # (batch, 2) — absolute field coords
        target_oren = u[:, 2:3]  # (batch, 1) — desired facing in [-pi, pi]
        kick_intent = u[:, 3]  # (batch,)

        # Clamp target to field bounds
        field_min = torch.tensor([-fc.half_length, -fc.half_width], device=u.device)
        field_max = torch.tensor([fc.half_length, fc.half_width], device=u.device)
        target_pos = target_pos.clamp(min=field_min, max=field_max)

        # --- Common state ---
        agent_pos = agent.state.pos
        agent_rot = agent.state.rot  # (batch, 1)
        has_ball = self._has_ball(agent)

        cos_rot = torch.cos(agent_rot.squeeze(-1))
        sin_rot = torch.sin(agent_rot.squeeze(-1))
        facing = torch.stack([cos_rot, sin_rot], dim=-1)

        # --- Navigation: PID toward target ---
        vel = self.pid_trans.calculate(agent_pos, target_pos, agent.name)

        # --- Orientation: PID toward explicit target_oren ---
        ang = self.pid_oren.calculate(agent_rot, target_oren, agent.name)

        # --- Kick state machine (2-frame dribbler release) ---
        to_target = target_pos - agent_pos
        dist_to_target = torch.norm(to_target, dim=-1, keepdim=True)
        to_target_dir = to_target / dist_to_target.clamp(min=1e-8)
        alignment = (facing * to_target_dir).sum(dim=-1)
        wants_kick = (kick_intent > dc.kick_intent_threshold) & has_ball & (alignment > dc.kick_align_threshold)

        pending = self.kick_pending[agent.name]

        # Frame 2: pending from last step AND still has ball + aligned → fire
        kick_fire = pending & has_ball & (alignment > dc.kick_align_threshold)

        # Frame 1: new kick request → set pending (dribbler off, no kick yet)
        new_pending = wants_kick & ~pending

        # Track kick events for reward function
        if not hasattr(self, "kick_fired"):
            self.kick_fired: dict[str, torch.Tensor] = {}
        self.kick_fired[agent.name] = kick_fire

        # Apply kick: set ball velocity along robot facing direction
        if kick_fire.any():
            kick_vel = facing * dc.kick_speed
            self.ball.state.vel = torch.where(
                kick_fire.unsqueeze(-1),
                kick_vel,
                self.ball.state.vel,
            )

        # Update pending state: fire clears, new request sets, else clear
        self.kick_pending[agent.name] = new_pending & ~kick_fire

        # --- Dribble: auto-engage when has_ball, not during kick sequence ---
        dribble_active = has_ball & ~pending & ~kick_fire
        if dribble_active.any():
            front_offset = facing * (fc.robot_radius + fc.ball_radius)
            dribble_target = agent_pos + front_offset
            attract_dir = dribble_target - self.ball.state.pos
            attract_force = attract_dir * dc.dribble_force
            self.ball.state.vel = (
                self.ball.state.vel + attract_force * dribble_active.unsqueeze(-1).float() * self.world.dt
            )

        # --- Write velocity commands for VelocityHolonomic ---
        agent.action.u = torch.cat([vel, ang], dim=-1)

    # ------------------------------------------------------------------
    # Macro-action processing (3D action space)
    # ------------------------------------------------------------------

    def _process_macro_action(self, agent: Agent):
        """Convert 3D [action_selector, target_x, target_y] to velocity commands.

        Decodes the macro-action type from action_selector, computes blended
        navigation/orientation targets per-env, then drives with BatchedPID.
        """
        dc = self.cfg.dynamics
        fc = self.cfg.field
        u = agent.action.u  # (batch, 3)

        # --- Decode macro-action ---
        action_selector = u[:, 0]  # [-1, 1]
        n = dc.n_macro_actions
        action_idx = ((action_selector + 1.0) * n / 2.0).long().clamp(0, n - 1)

        if hasattr(self, "current_macro_action"):
            self.current_macro_action[agent.name] = action_idx

        # --- Target position (absolute field coords, already scaled by u_multiplier) ---
        target_pos = u[:, 1:3]
        target_pos = target_pos.clamp(
            min=torch.tensor([-fc.half_length, -fc.half_width], device=u.device),
            max=torch.tensor([fc.half_length, fc.half_width], device=u.device),
        )

        # --- Common state ---
        agent_pos = agent.state.pos
        agent_rot = agent.state.rot  # (batch, 1)
        ball_pos = self.ball.state.pos
        has_ball = self._has_ball(agent)

        cos_rot = torch.cos(agent_rot.squeeze(-1))
        sin_rot = torch.sin(agent_rot.squeeze(-1))
        facing = torch.stack([cos_rot, sin_rot], dim=-1)

        # --- Action masks (per-env, exactly one is 1.0) ---
        is_go = (action_idx == MacroAction.GO_TO_BALL).unsqueeze(-1).float()
        is_kick = (action_idx == MacroAction.KICK_TO).unsqueeze(-1).float()
        is_drib = (action_idx == MacroAction.DRIBBLE_TO).unsqueeze(-1).float()
        is_move = (action_idx == MacroAction.MOVE_TO).unsqueeze(-1).float()

        # --- Blended navigation target ---
        # GO_TO_BALL → ball, KICK_TO → ball, DRIBBLE_TO → target/ball, MOVE_TO → target
        drib_nav = torch.where(has_ball.unsqueeze(-1), target_pos, ball_pos)
        nav_target = is_go * ball_pos + is_kick * ball_pos + is_drib * drib_nav + is_move * target_pos

        # --- Blended orientation target ---
        to_ball = ball_pos - agent_pos
        to_target = target_pos - agent_pos
        ball_oren = torch.atan2(to_ball[:, 1], to_ball[:, 0]).unsqueeze(-1)
        target_oren = torch.atan2(to_target[:, 1], to_target[:, 0]).unsqueeze(-1)
        drib_oren = torch.where(has_ball.unsqueeze(-1), target_oren, ball_oren)
        oren_target = is_go * ball_oren + is_kick * target_oren + is_drib * drib_oren + is_move * target_oren

        # --- Drive with BatchedPID (single call, state tracked internally) ---
        vel = self.pid_trans.calculate(agent_pos, nav_target, agent.name)
        ang = self.pid_oren.calculate(agent_rot, oren_target, agent.name)

        # KICK_TO with ball: stop translating (hold position while orienting)
        kick_hold = ((action_idx == MacroAction.KICK_TO) & has_ball).unsqueeze(-1)
        vel = torch.where(kick_hold, torch.zeros_like(vel), vel)

        # --- Apply ball side-effects (masked per macro) ---
        go_mask = action_idx == MacroAction.GO_TO_BALL
        kick_mask = action_idx == MacroAction.KICK_TO
        drib_mask = action_idx == MacroAction.DRIBBLE_TO

        # Kick (KICK_TO only): fire when aligned + has_ball
        to_target_flat = target_pos - agent_pos
        to_target_norm = to_target_flat / torch.norm(to_target_flat, dim=-1, keepdim=True).clamp(min=1e-8)
        alignment = (facing * to_target_norm).sum(dim=-1)
        kick_ready = kick_mask & has_ball & (alignment > dc.kick_align_threshold)

        if not hasattr(self, "kick_fired"):
            self.kick_fired: dict[str, torch.Tensor] = {}
        self.kick_fired[agent.name] = kick_ready

        if kick_ready.any():
            kick_ball_vel = facing * dc.kick_speed
            self.ball.state.vel = torch.where(
                kick_ready.unsqueeze(-1),
                kick_ball_vel,
                self.ball.state.vel,
            )

        # Dribble attract: GO_TO_BALL (when close), KICK_TO (hold while orienting),
        # DRIBBLE_TO (while moving)
        dribble_active = (go_mask & has_ball) | (kick_mask & has_ball & ~kick_ready) | (drib_mask & has_ball)
        if dribble_active.any():
            front_offset = facing * (fc.robot_radius + fc.ball_radius)
            dribble_target = agent_pos + front_offset
            attract_dir = dribble_target - ball_pos
            attract_force = attract_dir * dc.dribble_force
            self.ball.state.vel = (
                self.ball.state.vel + attract_force * dribble_active.unsqueeze(-1).float() * self.world.dt
            )

        # --- Write velocity commands for VelocityHolonomic ---
        agent.action.u = torch.cat([vel, ang], dim=-1)

    # ------------------------------------------------------------------
    # Legacy action processing (6D action space)
    # ------------------------------------------------------------------

    def _process_legacy_action(self, agent: Agent):
        """Convert 6D position-based actions to velocities via BatchedPID.

        Action priority: kick > turn_on_spot > move.
        Writes computed velocities into agent.action.u[:, 0:3] so that
        VelocityHolonomic can apply them as velocity commands.
        """
        dc = self.cfg.dynamics
        fc = self.cfg.field
        u = agent.action.u

        kick_action = u[:, 3]
        dribble_action = u[:, 4]
        turn_action = u[:, 5]
        target_oren = u[:, 2:3]  # (batch, 1)

        agent_pos = agent.state.pos
        ball_pos = self.ball.state.pos
        has_ball = self._has_ball(agent)

        # Facing direction
        cos_rot = torch.cos(agent.state.rot.squeeze(-1))
        sin_rot = torch.sin(agent.state.rot.squeeze(-1))
        facing = torch.stack([cos_rot, sin_rot], dim=-1)

        # --- Determine action mode per env ---
        kick_mode = (kick_action > 0) & has_ball
        turn_mode = (turn_action > 0) & has_ball & ~kick_mode

        # Track kick events for reward function (kick alignment shaping)
        if not hasattr(self, "kick_fired"):
            self.kick_fired: dict[str, torch.Tensor] = {}
        self.kick_fired[agent.name] = kick_mode

        # --- Compute MOVE velocities via BatchedPID ---
        delta_pos = u[:, 0:2]  # (batch, 2)
        target_pos = agent_pos + delta_pos
        field_min = torch.tensor([-fc.half_length, -fc.half_width], device=self.world.device)
        field_max = torch.tensor([fc.half_length, fc.half_width], device=self.world.device)
        target_pos = target_pos.clamp(min=field_min, max=field_max)

        move_vel = self.pid_trans.calculate(agent_pos, target_pos, agent.name)

        # Orientation via BatchedPID
        move_ang_vel = self.pid_oren.calculate(agent.state.rot, target_oren, agent.name)

        # --- Compute TURN_ON_SPOT velocities (ball pivot) ---
        turn_ang_vel = move_ang_vel  # same PID output for orientation

        # Perpendicular velocity for ball pivot (orbit around ball center)
        to_ball = ball_pos - agent_pos
        to_ball_dist = torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-6)
        to_ball_dir = to_ball / to_ball_dist
        perp_dir = torch.stack([-to_ball_dir[:, 1], to_ball_dir[:, 0]], dim=-1)
        turn_linear_vel = perp_dir * turn_ang_vel * to_ball_dist * dc.turn_on_spot_radius_modifier

        # Clamp turn linear velocity
        turn_speed = torch.norm(turn_linear_vel, dim=-1, keepdim=True)
        turn_scale = torch.where(
            turn_speed > dc.robot_max_speed,
            dc.robot_max_speed / turn_speed.clamp(min=1e-8),
            torch.ones_like(turn_speed),
        )
        turn_linear_vel = turn_linear_vel * turn_scale

        # --- Merge velocities based on mode ---
        zero_vel = torch.zeros_like(move_vel)
        zero_ang = torch.zeros_like(move_ang_vel)

        final_vel = torch.where(kick_mode.unsqueeze(-1), zero_vel, move_vel)
        final_vel = torch.where(turn_mode.unsqueeze(-1), turn_linear_vel, final_vel)

        final_ang = torch.where(kick_mode.unsqueeze(-1), zero_ang, move_ang_vel)
        final_ang = torch.where(turn_mode.unsqueeze(-1), turn_ang_vel, final_ang)

        # --- Overwrite action dims 0-2 with computed velocities ---
        agent.action.u = torch.cat(
            [
                final_vel,  # dims 0-1: linear velocity (global frame)
                final_ang,  # dim 2: angular velocity
                u[:, 3:],  # dims 3-5: pass through (kick, dribble, turn triggers)
            ],
            dim=-1,
        )

        # --- Kick: set ball velocity directly ---
        if kick_mode.any():
            kick_vel = facing * dc.kick_speed
            self.ball.state.vel = torch.where(
                kick_mode.unsqueeze(-1),
                kick_vel,
                self.ball.state.vel,
            )

        # --- Dribble: binary attract (move mode + turn mode both support dribble) ---
        dribble_active = (dribble_action > 0) & has_ball & ~kick_mode
        dribble_active = dribble_active | turn_mode
        if dribble_active.any():
            front_offset = facing * (fc.robot_radius + fc.ball_radius)
            dribble_target = agent_pos + front_offset
            attract_dir = dribble_target - ball_pos
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
                + has_ball(1) + active zone edge distances(4) + time_remaining(1) = 14
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

        # Has-ball flag (1D): 1 if this agent is within dribble range of ball
        has_ball = self._has_ball(agent).unsqueeze(-1).float()
        obs_parts.append(has_ball)

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

        # Normalized time remaining (1D)
        time_remaining = (1.0 - self.steps.float() / self.cfg.max_steps).unsqueeze(-1)
        obs_parts.append(time_remaining)

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
                self.prev_passer_facing_receiver_cos = self._compute_facing_ball_cos(agent, self.attackers[1].state.pos)
        if self.cfg.n_attackers >= 2 and agent is self.attackers[1]:
            self.prev_receiver_to_ball_dist = torch.norm(agent.state.pos - bp, dim=-1)
            self.prev_receiver_facing_ball_cos = self._compute_facing_ball_cos(agent, bp)

    # ------------------------------------------------------------------
    # Pass & interception tracking
    # ------------------------------------------------------------------

    def _update_pass_tracking(self):
        dc = self.cfg.dynamics
        ball_pos = self.ball.state.pos

        for i, agent in enumerate(self.attackers):
            dist = torch.norm(agent.state.pos - ball_pos, dim=-1)
            near_ball = dist < dc.dribble_dist_threshold
            new_pass = near_ball & (self.last_holder != i) & (self.last_holder >= 0)
            self.pass_completed = self.pass_completed | new_pass
            self.pass_count = self.pass_count + new_pass.long()
            self.last_holder = torch.where(near_ball, i, self.last_holder)

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
