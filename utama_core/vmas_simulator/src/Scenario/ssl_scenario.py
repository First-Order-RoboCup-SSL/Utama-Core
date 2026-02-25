import math

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from vmas.simulator.scenario import BaseScenario

from utama_core.vmas_simulator.src.Scenario.rewards import compute_team_reward
from utama_core.vmas_simulator.src.Utils.config import SSLFieldConfig, SSLScenarioConfig


class SSLScenario(BaseScenario):
    """RoboCup SSL scenario for VMAS.

    Creates a Division B field (9m x 6m) with configurable numbers of
    blue and yellow robots plus a ball. Uses the SSL standard coordinate
    system (like GRSim): X along field length, Y along width, orientation
    counter-clockwise from X-axis in radians.

    Agents' action space: 5 continuous values
      [0] force_x   -- HolonomicWithRotation force x
      [1] force_y   -- HolonomicWithRotation force y
      [2] torque    -- HolonomicWithRotation torque
      [3] kick      -- thresholded > 0 triggers kick impulse
      [4] dribble   -- thresholded > 0 activates dribbler
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.cfg: SSLScenarioConfig = kwargs.get("scenario_config", SSLScenarioConfig())
        fc = self.cfg.field_config
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

        # --- Blue Team ---
        self.blue_agents: list[Agent] = []
        for i in range(self.cfg.n_blue):
            agent = Agent(
                name=f"blue_{i}",
                shape=Sphere(radius=fc.robot_radius),
                movable=True,
                rotatable=True,
                collide=True,
                mass=dc.robot_mass,
                max_speed=dc.robot_max_speed,
                color=torch.tensor([0.2, 0.2, 0.9]),
                dynamics=HolonomicWithRotation(),
                action_size=5,
                u_multiplier=[
                    dc.robot_max_speed,
                    dc.robot_max_speed,
                    dc.robot_max_angular_vel,
                    1.0,
                    1.0,
                ],
                linear_friction=dc.robot_friction,
            )
            world.add_agent(agent)
            self.blue_agents.append(agent)

        # --- Yellow Team ---
        self.yellow_agents: list[Agent] = []
        for i in range(self.cfg.n_yellow):
            agent = Agent(
                name=f"yellow_{i}",
                shape=Sphere(radius=fc.robot_radius),
                movable=True,
                rotatable=True,
                collide=True,
                mass=dc.robot_mass,
                max_speed=dc.robot_max_speed,
                color=torch.tensor([0.9, 0.9, 0.2]),
                dynamics=HolonomicWithRotation(),
                action_size=5,
                u_multiplier=[
                    dc.robot_max_speed,
                    dc.robot_max_speed,
                    dc.robot_max_angular_vel,
                    1.0,
                    1.0,
                ],
                linear_friction=dc.robot_friction,
            )
            world.add_agent(agent)
            self.yellow_agents.append(agent)

        # --- Ball (movable Landmark) ---
        self.ball = Landmark(
            name="ball",
            shape=Sphere(radius=fc.ball_radius),
            movable=True,
            rotatable=False,
            collide=True,
            mass=dc.ball_mass,
            color=torch.tensor([1.0, 0.5, 0.0]),
            linear_friction=dc.ball_friction,
            drag=dc.ball_drag,
        )
        world.add_landmark(self.ball)

        # --- Boundary Walls ---
        self._create_boundary_walls(world, fc)

        # --- Goals ---
        self._create_goals(world, fc)

        # --- State Tracking ---
        self.goal_scored_blue = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.goal_scored_yellow = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.steps = torch.zeros(batch_dim, device=device, dtype=torch.long)

        # --- Kick Cooldown (per agent) ---
        self.kick_cooldown: dict[str, torch.Tensor] = {}
        for agent in self.blue_agents + self.yellow_agents:
            self.kick_cooldown[agent.name] = torch.zeros(batch_dim, device=device, dtype=torch.long)

        return world

    def _create_boundary_walls(self, world: World, fc: SSLFieldConfig):
        """Create boundary walls with gaps for goals."""
        wall_thickness = 0.05
        self.walls: list[tuple[str, Landmark]] = []

        # Top and bottom full-width walls
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

        # Left and right walls: two segments each (above and below goal)
        half_wall_height = (fc.half_width - fc.goal_width / 2) / 2
        for side, x_sign in [("left", -1), ("right", 1)]:
            for seg, y_sign in [("top", 1), ("bottom", -1)]:
                y_center = y_sign * (fc.goal_width / 2 + half_wall_height)
                wall = Landmark(
                    name=f"wall_{side}_{seg}",
                    shape=Box(length=wall_thickness, width=half_wall_height * 2),
                    movable=False,
                    collide=True,
                    color=torch.tensor([0.3, 0.6, 0.3]),
                )
                world.add_landmark(wall)
                self.walls.append((f"{side}_{seg}", wall, x_sign * fc.half_length, y_center))

    def _create_goals(self, world: World, fc: SSLFieldConfig):
        """Create goal back-walls."""
        self.left_goal_back = Landmark(
            name="goal_left_back",
            shape=Box(length=0.02, width=fc.goal_width),
            movable=False,
            collide=True,
            color=torch.tensor([0.8, 0.8, 0.8]),
        )
        world.add_landmark(self.left_goal_back)

        self.right_goal_back = Landmark(
            name="goal_right_back",
            shape=Box(length=0.02, width=fc.goal_width),
            movable=False,
            collide=True,
            color=torch.tensor([0.8, 0.8, 0.8]),
        )
        world.add_landmark(self.right_goal_back)

    def reset_world_at(self, env_index: int = None):
        fc = self.cfg.field_config

        # Blue team starts on left (facing right), matching LEFT_START_ONE from formations.py
        blue_formations = [
            (-4.2, 0.0, 0.0),
            (-3.4, 0.2, 0.0),
            (-3.4, -0.2, 0.0),
            (-0.7, 0.0, 0.0),
            (-0.7, -2.25, 0.0),
            (-0.7, 2.25, 0.0),
        ]
        # Yellow team starts on right (facing left), matching RIGHT_START_ONE
        yellow_formations = [
            (4.2, 0.0, math.pi),
            (3.4, -0.2, math.pi),
            (3.4, 0.2, math.pi),
            (0.7, 0.0, math.pi),
            (0.7, 2.25, math.pi),
            (0.7, -2.25, math.pi),
        ]

        self._reset_agents(self.blue_agents, blue_formations, env_index)
        self._reset_agents(self.yellow_agents, yellow_formations, env_index)

        # Ball at center
        self._set_entity_state(self.ball, 0.0, 0.0, env_index)

        # Position walls and goals
        for _, wall, wx, wy in self.walls:
            self._set_entity_state(wall, wx, wy, env_index, movable=False)

        left_goal_x = -(fc.half_length + fc.goal_depth / 2)
        right_goal_x = fc.half_length + fc.goal_depth / 2
        self._set_entity_state(self.left_goal_back, left_goal_x, 0.0, env_index, movable=False)
        self._set_entity_state(self.right_goal_back, right_goal_x, 0.0, env_index, movable=False)

        # Reset tracking
        if env_index is None:
            self.goal_scored_blue.zero_()
            self.goal_scored_yellow.zero_()
            self.steps.zero_()
            for name in self.kick_cooldown:
                self.kick_cooldown[name].zero_()
        else:
            self.goal_scored_blue[env_index] = False
            self.goal_scored_yellow[env_index] = False
            self.steps[env_index] = 0
            for name in self.kick_cooldown:
                self.kick_cooldown[name][env_index] = 0

    def _reset_agents(self, agents: list[Agent], formations: list[tuple], env_index: int = None):
        for i, agent in enumerate(agents):
            if i < len(formations):
                x, y, theta = formations[i]
            else:
                x, y, theta = -1.0 - 0.3 * i, 0.0, 0.0

            pos = torch.tensor([x, y], device=self.world.device, dtype=torch.float32)
            rot = torch.tensor([theta], device=self.world.device, dtype=torch.float32)
            zero2 = torch.zeros(2, device=self.world.device, dtype=torch.float32)

            if env_index is None:
                agent.set_pos(pos.unsqueeze(0).expand(self.world.batch_dim, -1), batch_index=None)
                agent.set_vel(zero2.unsqueeze(0).expand(self.world.batch_dim, -1), batch_index=None)
                agent.set_rot(rot.unsqueeze(0).expand(self.world.batch_dim, -1), batch_index=None)
            else:
                agent.set_pos(pos, batch_index=env_index)
                agent.set_vel(zero2, batch_index=env_index)
                agent.set_rot(rot, batch_index=env_index)

    def _set_entity_state(self, entity, x: float, y: float, env_index: int = None, movable: bool = True):
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

    def process_action(self, agent: Agent):
        """Legacy method — calls both kick and dribble. Prefer calling them separately."""
        dribble_active = self.compute_dribble_state(agent)
        kick_triggered = self.process_kick(agent)
        # Kick overrides dribble: if kick fired, suppress dribble re-lock
        dribble_active = dribble_active & ~kick_triggered
        self.process_dribble(agent, dribble_active)

    def process_kick(self, agent: Agent) -> torch.Tensor:
        """Apply kick action: set ball velocity directly with momentum preservation.

        Must be called BEFORE world.step() so the ball is still in range
        (collision forces haven't pushed it away yet) and so that world.step()
        applies friction to the kicked ball velocity.

        Physics matched to rSim:
        - Direct velocity set (not impulse)
        - Preserves kick_damp_factor of existing ball momentum along kick direction
        - 10-step cooldown between kicks

        Returns:
            kick_triggered: (batch_dim,) bool mask of envs where kick fired
        """
        kick_action = agent.action.u[:, 3]  # (batch_dim,)

        agent_pos = agent.state.pos
        ball_pos = self.ball.state.pos
        dc = self.cfg.dynamics
        fc = self.cfg.field_config

        dist_to_ball = torch.norm(agent_pos - ball_pos, dim=-1)
        in_range = dist_to_ball < dc.dribble_dist_threshold

        cos_rot = torch.cos(agent.state.rot.squeeze(-1))
        sin_rot = torch.sin(agent.state.rot.squeeze(-1))
        kick_dir = torch.stack([cos_rot, sin_rot], dim=-1)

        cooldown = self.kick_cooldown[agent.name]
        kick_ready = cooldown <= 0
        kick_triggered = (kick_action > 0) & in_range & kick_ready

        if kick_triggered.any():
            kick_speed_val = kick_action.clamp(min=0.0, max=dc.kick_speed)

            ball_vel = self.ball.state.vel
            existing_along_kick = (ball_vel * kick_dir).sum(dim=-1, keepdim=True)

            new_ball_vel = (
                kick_dir * kick_speed_val.unsqueeze(-1) + kick_dir * existing_along_kick * dc.kick_damp_factor
            )

            mask = kick_triggered.unsqueeze(-1).float()
            self.ball.state.vel = self.ball.state.vel * (1 - mask) + new_ball_vel * mask

            # Move ball just outside robot to prevent VMAS collision forces
            # from adding extra velocity during world.step()
            separation = kick_dir * (fc.robot_radius + fc.ball_radius + 0.005)
            kicked_pos = agent_pos + separation
            self.ball.state.pos = self.ball.state.pos * (1 - mask) + kicked_pos * mask

            cooldown = torch.where(
                kick_triggered,
                torch.full_like(cooldown, dc.kick_cooldown_steps),
                cooldown,
            )

        # Decrement cooldown (uses updated cooldown, not stale local var)
        self.kick_cooldown[agent.name] = (cooldown - 1).clamp(min=0)

        return kick_triggered

    def compute_dribble_state(self, agent: Agent) -> torch.Tensor:
        """Check if dribble should be active. Call BEFORE world.step().

        Returns a boolean mask (batch_dim,) indicating dribble eligibility.
        Must be called before collision forces push the ball away.
        """
        dribble_action = agent.action.u[:, 4]
        dist_to_ball = torch.norm(agent.state.pos - self.ball.state.pos, dim=-1)
        in_range = dist_to_ball < self.cfg.dynamics.dribble_dist_threshold
        return (dribble_action > 0) & in_range

    def process_dribble(self, agent: Agent, dribble_active: torch.Tensor):
        """Position-lock ball to kicker face (matches rSim hinge joint).

        Must be called AFTER world.step() so it overrides collision forces.
        Uses the pre-computed dribble_active mask from compute_dribble_state().
        """
        if not dribble_active.any():
            return

        fc = self.cfg.field_config
        cos_rot = torch.cos(agent.state.rot.squeeze(-1))
        sin_rot = torch.sin(agent.state.rot.squeeze(-1))
        kick_dir = torch.stack([cos_rot, sin_rot], dim=-1)

        front_offset = kick_dir * (fc.robot_radius + fc.ball_radius)
        target_pos = agent.state.pos + front_offset

        mask = dribble_active.unsqueeze(-1).float()
        self.ball.state.pos = self.ball.state.pos * (1 - mask) + target_pos * mask
        self.ball.state.vel = self.ball.state.vel * (1 - mask) + agent.state.vel * mask

    def observation(self, agent: Agent) -> torch.Tensor:
        """Per-agent ego-centric observation vector. Shape: (batch_dim, obs_size).

        Contents (all positions relative to agent):
          - Own velocity (2) + angular velocity (1) + rotation (1) = 4
          - Has ball (1) — 1.0 if ball within dribble threshold, else 0.0
          - Ball relative position (2) + ball velocity (2) = 4
          - Per teammate (n_team - 1): relative position (2)
          - Per opponent (n_opp): relative position (2)
          - Own goal relative position (2)
          - Opponent goal relative position (2)
        """
        obs_parts: list[torch.Tensor] = []

        # Own state
        obs_parts.append(agent.state.vel)  # (batch, 2)
        if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
            obs_parts.append(agent.state.ang_vel)  # (batch, 1)
        else:
            obs_parts.append(torch.zeros(self.world.batch_dim, 1, device=self.world.device))
        obs_parts.append(agent.state.rot)  # (batch, 1)

        # Has ball (binary: ball within dribble threshold)
        dist_to_ball = torch.norm(agent.state.pos - self.ball.state.pos, dim=-1, keepdim=True)
        has_ball = (dist_to_ball < self.cfg.dynamics.dribble_dist_threshold).float()
        obs_parts.append(has_ball)  # (batch, 1)

        # Ball relative
        ball_rel_pos = self.ball.state.pos - agent.state.pos
        obs_parts.append(ball_rel_pos)
        if self.cfg.observe_ball_velocity:
            obs_parts.append(self.ball.state.vel)

        # Team membership
        is_blue = agent in self.blue_agents
        teammates = self.blue_agents if is_blue else self.yellow_agents
        opponents = self.yellow_agents if is_blue else self.blue_agents

        # Teammate relative positions
        if self.cfg.observe_teammates:
            for t in teammates:
                if t is agent:
                    continue
                obs_parts.append(t.state.pos - agent.state.pos)

        # Opponent relative positions
        if self.cfg.observe_opponents:
            for o in opponents:
                obs_parts.append(o.state.pos - agent.state.pos)

        # Goal positions (relative)
        fc = self.cfg.field_config
        if is_blue:
            own_goal = torch.tensor([-fc.half_length, 0.0], device=self.world.device)
            opp_goal = torch.tensor([fc.half_length, 0.0], device=self.world.device)
        else:
            own_goal = torch.tensor([fc.half_length, 0.0], device=self.world.device)
            opp_goal = torch.tensor([-fc.half_length, 0.0], device=self.world.device)

        obs_parts.append(own_goal.unsqueeze(0).expand(self.world.batch_dim, -1) - agent.state.pos)
        obs_parts.append(opp_goal.unsqueeze(0).expand(self.world.batch_dim, -1) - agent.state.pos)

        return torch.cat(obs_parts, dim=-1)

    def reward(self, agent: Agent) -> torch.Tensor:
        """Compute reward for an agent. Shape: (batch_dim,).

        Blue team attacks right goal (+x), yellow attacks left goal (-x).
        """
        is_blue = agent in self.blue_agents
        return compute_team_reward(
            world=self.world,
            ball=self.ball,
            blue_agents=self.blue_agents,
            yellow_agents=self.yellow_agents,
            is_blue=is_blue,
            cfg=self.cfg,
            goal_scored_blue=self.goal_scored_blue,
            goal_scored_yellow=self.goal_scored_yellow,
        )

    def done(self) -> torch.Tensor:
        """Episode ends when a goal is scored or max steps reached."""
        self.steps += 1

        fc = self.cfg.field_config
        ball_x = self.ball.state.pos[:, 0]
        ball_y = self.ball.state.pos[:, 1]

        # Goal detection
        in_right_goal = (ball_x > fc.half_length) & (ball_y.abs() < fc.goal_width / 2)
        in_left_goal = (ball_x < -fc.half_length) & (ball_y.abs() < fc.goal_width / 2)

        self.goal_scored_blue = self.goal_scored_blue | in_right_goal
        self.goal_scored_yellow = self.goal_scored_yellow | in_left_goal

        goal_scored = in_right_goal | in_left_goal
        timeout = self.steps >= self.cfg.max_steps

        return goal_scored | timeout

    def info(self, agent: Agent) -> dict:
        is_blue = agent in self.blue_agents
        return {
            "goal_scored_by_team": self.goal_scored_blue if is_blue else self.goal_scored_yellow,
            "goal_conceded_by_team": self.goal_scored_yellow if is_blue else self.goal_scored_blue,
            "steps": self.steps,
        }
