"""VmasSSL: VMAS-backed SSL simulator. Mirrors the RSimSSL interface.

Physics matched to rSim (rSoccer's C++ ODE engine):
- Ball friction: Coulomb model (mu * g deceleration) + linear damping
- Kick: direct velocity set with momentum preservation
- Dribble: position-lock ball to kicker face
- Kick cooldown: 10-step lockout
- Ball bounce: pseudo-restitution on wall/robot contact
- Ball dead zone: stop below 0.01 m/s
- Robot acceleration limiting
- Ball push on robot contact (dribbler off)
"""

import math
from typing import List

import torch

from utama_core.rsoccer_simulator.src.Entities import (
    Ball,
    Field,
    Frame,
    FrameSSL,
    Robot,
)
from utama_core.vmas_simulator.src.Scenario.ssl_scenario import SSLScenario
from utama_core.vmas_simulator.src.Utils.config import SSLScenarioConfig


class VmasSSL:
    """VMAS-backed SSL simulator. Mirrors RSimSSL interface.

    For RL training: use with num_envs > 1
    For BT comparison (StrategyRunner): use with num_envs = 1

    Uses SSL standard coordinate system (like GRSim):
    - X along field length, Y along field width
    - Orientation: radians, counter-clockwise from X-axis
    - No Y-axis inversion (unlike RSim)
    """

    def __init__(
        self,
        field_type: int = 1,
        n_robots_blue: int = 6,
        n_robots_yellow: int = 6,
        time_step_ms: int = 16,
        num_envs: int = 1,
        device: str = "cpu",
        scenario_config: SSLScenarioConfig = None,
    ):
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.num_envs = num_envs
        self.device = device

        if scenario_config is None:
            scenario_config = SSLScenarioConfig(
                n_blue=n_robots_blue,
                n_yellow=n_robots_yellow,
            )
        self.scenario_config = scenario_config

        self.scenario = SSLScenario()
        self.world = self.scenario.make_world(
            batch_dim=num_envs,
            device=torch.device(device),
            scenario_config=scenario_config,
        )
        # BaseScenario.world property requires _world to be set.
        # Normally make_env() does this, but we call make_world() directly.
        self.scenario._world = self.world
        self.scenario.reset_world_at(env_index=None)

        self.field_config = scenario_config.field_config

        # Acceleration limiting state
        dc = scenario_config.dynamics
        self._prev_vel: dict[tuple[bool, int], tuple[float, float, float]] = {}
        self._max_accel = dc.robot_max_acceleration
        self._max_angular_accel = dc.robot_max_angular_acceleration

        # Ball possession tracking (set during send_commands, read by get_frame)
        self._has_ball: dict[str, bool] = {}

    def reset(self, frame: Frame):
        """Reset using a Frame object (matches RSimSSL.reset interface)."""
        self._prev_vel.clear()
        self._has_ball.clear()
        # Reset kick cooldowns
        for name in self.scenario.kick_cooldown:
            self.scenario.kick_cooldown[name].zero_()
        # Convert Frame positions to tensor and set them
        for robot_id, robot in frame.robots_blue.items():
            if robot_id < self.n_robots_blue:
                agent = self.scenario.blue_agents[robot_id]
                pos = torch.tensor([robot.x, robot.y], device=self.device, dtype=torch.float32)
                # Frame stores theta in degrees (rsoccer convention); convert to radians
                theta_rad = math.radians(robot.theta) if robot.theta is not None else 0.0
                rot = torch.tensor([theta_rad], device=self.device, dtype=torch.float32)
                agent.set_pos(pos.unsqueeze(0).expand(self.num_envs, -1), batch_index=None)
                agent.set_rot(rot.unsqueeze(0).expand(self.num_envs, -1), batch_index=None)
                agent.set_vel(
                    torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32),
                    batch_index=None,
                )

        for robot_id, robot in frame.robots_yellow.items():
            if robot_id < self.n_robots_yellow:
                agent = self.scenario.yellow_agents[robot_id]
                pos = torch.tensor([robot.x, robot.y], device=self.device, dtype=torch.float32)
                theta_rad = math.radians(robot.theta) if robot.theta is not None else 0.0
                rot = torch.tensor([theta_rad], device=self.device, dtype=torch.float32)
                agent.set_pos(pos.unsqueeze(0).expand(self.num_envs, -1), batch_index=None)
                agent.set_rot(rot.unsqueeze(0).expand(self.num_envs, -1), batch_index=None)
                agent.set_vel(
                    torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32),
                    batch_index=None,
                )

        if frame.ball is not None and frame.ball.x is not None:
            ball_pos = torch.tensor([frame.ball.x, frame.ball.y], device=self.device, dtype=torch.float32)
            ball_vel = torch.tensor([frame.ball.v_x, frame.ball.v_y], device=self.device, dtype=torch.float32)
            self.scenario.ball.set_pos(ball_pos.unsqueeze(0).expand(self.num_envs, -1), batch_index=None)
            self.scenario.ball.set_vel(ball_vel.unsqueeze(0).expand(self.num_envs, -1), batch_index=None)

    def send_commands(self, commands: List[Robot]):
        """Convert Robot commands to VMAS state updates and step the world.

        Commands are velocity commands in SSL standard frame (like GRSim).
        Velocity-based control with acceleration rate limiting to match rSim.
        """
        dt = self.world.dt
        dc = self.scenario.cfg.dynamics

        # Build dribbler state lookup for ball push logic
        dribbler_on: dict[tuple[bool, int], bool] = {}
        for cmd in commands:
            dribbler_on[(cmd.yellow, cmd.id)] = bool(cmd.dribbler)

        # Apply velocity commands with acceleration limiting
        for cmd in commands:
            if cmd.yellow:
                if cmd.id >= self.n_robots_yellow:
                    continue
                agent = self.scenario.yellow_agents[cmd.id]
            else:
                if cmd.id >= self.n_robots_blue:
                    continue
                agent = self.scenario.blue_agents[cmd.id]

            # Convert local velocity commands to global frame
            rot = agent.state.rot[0, 0].item()
            cos_r = math.cos(rot)
            sin_r = math.sin(rot)
            local_vx = float(cmd.v_x) if cmd.v_x else 0.0
            local_vy = float(cmd.v_y) if cmd.v_y else 0.0
            target_global_vx = local_vx * cos_r - local_vy * sin_r
            target_global_vy = local_vx * sin_r + local_vy * cos_r
            target_v_theta = float(cmd.v_theta) if cmd.v_theta else 0.0

            # Acceleration rate limiting
            key = (cmd.yellow, cmd.id)
            prev_vx, prev_vy, prev_vt = self._prev_vel.get(key, (0.0, 0.0, 0.0))

            # Linear acceleration limiting
            dvx = target_global_vx - prev_vx
            dvy = target_global_vy - prev_vy
            dv_norm = math.hypot(dvx, dvy)
            max_dv = self._max_accel * dt
            if dv_norm > max_dv and dv_norm > 0:
                scale = max_dv / dv_norm
                dvx *= scale
                dvy *= scale
            global_vx = prev_vx + dvx
            global_vy = prev_vy + dvy

            # Angular acceleration limiting
            dvt = target_v_theta - prev_vt
            max_dvt = self._max_angular_accel * dt
            dvt = max(-max_dvt, min(max_dvt, dvt))
            v_theta = prev_vt + dvt

            self._prev_vel[key] = (global_vx, global_vy, v_theta)

            # Set velocity
            vel = torch.tensor([[global_vx, global_vy]], device=self.device, dtype=torch.float32).expand(
                self.num_envs, -1
            )
            agent.set_vel(vel, batch_index=None)

            # Angular velocity
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
                ang_vel = torch.tensor([[v_theta]], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
                agent.state.ang_vel = ang_vel

            # Update rotation by integrating angular velocity
            new_rot = agent.state.rot + v_theta * dt
            agent.set_rot(new_rot, batch_index=None)

        # Ball push: transfer momentum from robot contact (dribbler off)
        self._apply_robot_ball_push(commands, dribbler_on)

        # Pre-step: set up action tensors, compute dribble state, and process kicks
        # BEFORE world.step() so the ball is still in range (collision forces
        # haven't pushed it away) and world.step() applies friction to kicked velocity.
        dribble_masks: dict[str, torch.Tensor] = {}
        agents_with_actions: list = []
        for cmd in commands:
            if cmd.yellow:
                if cmd.id >= self.n_robots_yellow:
                    continue
                agent = self.scenario.yellow_agents[cmd.id]
            else:
                if cmd.id >= self.n_robots_blue:
                    continue
                agent = self.scenario.blue_agents[cmd.id]

            kick_val = float(cmd.kick_v_x) if cmd.kick_v_x else 0.0
            dribble_val = 1.0 if cmd.dribbler else 0.0
            action_u = torch.zeros(self.num_envs, 5, device=self.device, dtype=torch.float32)
            action_u[:, 3] = kick_val
            action_u[:, 4] = dribble_val
            agent.action.u = action_u

            # Compute dribble eligibility before collision forces (pre-step)
            dribble_masks[agent.name] = self.scenario.compute_dribble_state(agent)

            # Kick before world.step() (ball still in range)
            kick_triggered = self.scenario.process_kick(agent)

            # Kick overrides dribble: if kick fired, suppress dribble re-lock
            # so the ball actually leaves the robot instead of being snapped back
            if kick_triggered.any():
                dribble_masks[agent.name] = dribble_masks[agent.name] & ~kick_triggered

            agents_with_actions.append(agent)

        # Step the world physics (handles collisions, friction, integration)
        self.world.step()

        # Post-step: dribble position-lock using pre-step range check
        # and update ball possession tracking
        dc = self.scenario.cfg.dynamics
        ball_pos = self.scenario.ball.state.pos
        for agent in agents_with_actions:
            dribble_active = dribble_masks[agent.name]
            self.scenario.process_dribble(agent, dribble_active)

            # Track ball possession: dribbling OR within contact distance
            dist = torch.norm(agent.state.pos - ball_pos, dim=-1)
            in_contact = dist < dc.dribble_dist_threshold
            self._has_ball[agent.name] = bool(dribble_active.any().item() or in_contact.any().item())

        # Post-step: ball dead zone, wall/robot bounce
        self._apply_ball_post_processing()

    def _apply_robot_ball_push(
        self,
        commands: List[Robot],
        dribbler_on: dict[tuple[bool, int], bool],
    ):
        """Transfer robot velocity to ball on contact when dribbler is off.

        Since send_commands() sets velocities directly (bypassing physics forces),
        this provides the momentum transfer that rSim gets from ODE collisions.
        When the robot stops, the ball retains its velocity and rolls away.
        """
        dc = self.scenario.cfg.dynamics
        fc = self.scenario.cfg.field_config
        ball_pos = self.scenario.ball.state.pos  # (num_envs, 2)
        ball_vel = self.scenario.ball.state.vel
        contact_dist = fc.robot_radius + fc.ball_radius + 0.005  # small margin

        for agent in self.scenario.blue_agents + self.scenario.yellow_agents:
            # Check if dribbler is on for this agent
            is_yellow = agent in self.scenario.yellow_agents
            agent_idx = (
                self.scenario.yellow_agents.index(agent) if is_yellow else self.scenario.blue_agents.index(agent)
            )
            key = (is_yellow, agent_idx)
            if dribbler_on.get(key, False):
                continue  # Dribbler handles ball, skip push

            agent_pos = agent.state.pos
            agent_vel = agent.state.vel

            delta = ball_pos - agent_pos
            dist = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
            in_contact = dist.squeeze(-1) < contact_dist

            if not in_contact.any():
                continue

            # Normal direction: agent → ball
            normal = delta / dist

            # Relative velocity along contact normal (robot closing speed on ball)
            robot_vel_along = (agent_vel * normal).sum(dim=-1, keepdim=True)
            ball_vel_along = (ball_vel * normal).sum(dim=-1, keepdim=True)
            rel_vel_along = robot_vel_along - ball_vel_along

            # Only push if robot is closing on ball (moving faster than ball)
            pushing = (rel_vel_along > 0) & in_contact.unsqueeze(-1)

            if pushing.any():
                # Transfer relative velocity along normal with coefficient
                push_vel = normal * rel_vel_along * dc.ball_push_transfer_coeff
                self.scenario.ball.state.vel = ball_vel + push_vel * pushing.float()
                ball_vel = self.scenario.ball.state.vel  # Update for next agent

    def _apply_ball_post_processing(self):
        """Apply ball dead zone and pseudo-restitution after world.step().

        rSim features not natively supported by VMAS:
        - Dead zone: stop ball below 0.01 m/s
        - Wall bounce: reflect velocity with restitution coefficient
        - Robot-ball bounce: reflect relative velocity on contact
        """
        dc = self.scenario.cfg.dynamics
        fc = self.scenario.cfg.field_config
        ball = self.scenario.ball
        ball_vel = ball.state.vel
        ball_pos = ball.state.pos

        # --- Ball speed dead zone ---
        ball_speed = torch.norm(ball_vel, dim=-1, keepdim=True)
        dead = ball_speed < dc.ball_speed_dead_zone
        ball_vel = torch.where(dead, torch.zeros_like(ball_vel), ball_vel)

        # --- Wall bounce (pseudo-restitution) ---
        ball_r = fc.ball_radius
        margin = 0.01  # Detection margin for wall proximity

        # Top/bottom walls
        at_top = ball_pos[:, 1:2] >= (fc.half_width - ball_r - margin)
        at_bottom = ball_pos[:, 1:2] <= -(fc.half_width - ball_r - margin)
        y_into_top = at_top & (ball_vel[:, 1:2] > 0)
        y_into_bottom = at_bottom & (ball_vel[:, 1:2] < 0)
        y_reflect = y_into_top | y_into_bottom
        y_speed_ok = ball_vel[:, 1:2].abs() > dc.ball_bounce_vel_threshold
        y_bounce = y_reflect & y_speed_ok
        ball_vel_y = ball_vel[:, 1:2].clone()
        ball_vel_y = torch.where(y_bounce, -ball_vel_y * dc.ball_restitution, ball_vel_y)
        ball_vel_y = torch.where(y_reflect & ~y_speed_ok, torch.zeros_like(ball_vel_y), ball_vel_y)

        # Left/right walls (skip goal opening)
        in_goal_y = ball_pos[:, 1:2].abs() < (fc.goal_width / 2)
        at_left = (ball_pos[:, 0:1] <= -(fc.half_length - ball_r - margin)) & ~in_goal_y
        at_right = (ball_pos[:, 0:1] >= (fc.half_length - ball_r - margin)) & ~in_goal_y
        x_into_left = at_left & (ball_vel[:, 0:1] < 0)
        x_into_right = at_right & (ball_vel[:, 0:1] > 0)
        x_reflect = x_into_left | x_into_right
        x_speed_ok = ball_vel[:, 0:1].abs() > dc.ball_bounce_vel_threshold
        x_bounce = x_reflect & x_speed_ok
        ball_vel_x = ball_vel[:, 0:1].clone()
        ball_vel_x = torch.where(x_bounce, -ball_vel_x * dc.ball_restitution, ball_vel_x)
        ball_vel_x = torch.where(x_reflect & ~x_speed_ok, torch.zeros_like(ball_vel_x), ball_vel_x)

        ball_vel = torch.cat([ball_vel_x, ball_vel_y], dim=-1)

        # --- Robot-ball bounce ---
        min_dist = fc.robot_radius + ball_r
        for agent in self.scenario.blue_agents + self.scenario.yellow_agents:
            delta = ball_pos - agent.state.pos
            dist = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
            in_contact = dist < (min_dist + 0.005)

            if not in_contact.any():
                continue

            normal = delta / dist
            rel_vel = ball_vel - agent.state.vel
            vel_along_normal = (rel_vel * normal).sum(dim=-1, keepdim=True)

            # Only bounce if approaching
            approaching = vel_along_normal < 0
            should_bounce = in_contact & approaching

            if should_bounce.any():
                bounce_dv = -(1 + dc.ball_restitution) * vel_along_normal * normal
                ball_vel = ball_vel + bounce_dv * should_bounce.float()

        ball.state.vel = ball_vel

    def get_frame(self) -> FrameSSL:
        """Read VMAS state and return a FrameSSL (env_index=0 for single-env mode).

        Returns positions in SSL standard coordinate system.
        """
        frame = FrameSSL()

        # Ball
        ball_pos = self.scenario.ball.state.pos[0]  # First env
        ball_vel = self.scenario.ball.state.vel[0]
        frame.ball = Ball(
            x=ball_pos[0].item(),
            y=ball_pos[1].item(),
            z=0.0,
            v_x=ball_vel[0].item(),
            v_y=ball_vel[1].item(),
        )

        # Blue robots
        for i, agent in enumerate(self.scenario.blue_agents):
            pos = agent.state.pos[0]
            vel = agent.state.vel[0]
            rot = agent.state.rot[0]

            # Ball possession from native tracking (set in send_commands)
            has_ball = self._has_ball.get(agent.name, False)

            # Angular velocity: rad/s → deg/s (matches Robot.theta being in degrees)
            ang_vel_deg = 0.0
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
                ang_vel_deg = math.degrees(agent.state.ang_vel[0, 0].item())

            robot = Robot(
                yellow=False,
                id=i,
                x=pos[0].item(),
                y=pos[1].item(),
                theta=math.degrees(rot[0].item()),
                v_x=vel[0].item(),
                v_y=vel[1].item(),
                v_theta=ang_vel_deg,
                infrared=has_ball,
            )
            frame.robots_blue[i] = robot

        # Yellow robots
        for i, agent in enumerate(self.scenario.yellow_agents):
            pos = agent.state.pos[0]
            vel = agent.state.vel[0]
            rot = agent.state.rot[0]

            has_ball = self._has_ball.get(agent.name, False)

            ang_vel_deg = 0.0
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
                ang_vel_deg = math.degrees(agent.state.ang_vel[0, 0].item())

            robot = Robot(
                yellow=True,
                id=i,
                x=pos[0].item(),
                y=pos[1].item(),
                theta=math.degrees(rot[0].item()),
                v_x=vel[0].item(),
                v_y=vel[1].item(),
                v_theta=ang_vel_deg,
                infrared=has_ball,
            )
            frame.robots_yellow[i] = robot

        return frame

    def get_field_params(self) -> Field:
        """Return Field dataclass from config."""
        fc = self.field_config
        return Field(
            length=fc.half_length * 2,
            width=fc.half_width * 2,
            penalty_length=fc.defense_area_length,
            penalty_width=fc.defense_area_width,
            goal_width=fc.goal_width,
            goal_depth=fc.goal_depth,
            ball_radius=fc.ball_radius,
            rbt_distance_center_kicker=0.075,
            rbt_kicker_thickness=0.005,
            rbt_kicker_width=0.08,
            rbt_wheel0_angle=60.0,
            rbt_wheel1_angle=135.0,
            rbt_wheel2_angle=225.0,
            rbt_wheel3_angle=300.0,
            rbt_radius=fc.robot_radius,
            rbt_wheel_radius=0.02,
            rbt_motor_max_rpm=1400.0,
        )

    def stop(self):
        """Cleanup (no subprocess to kill unlike RSim)."""
        pass
