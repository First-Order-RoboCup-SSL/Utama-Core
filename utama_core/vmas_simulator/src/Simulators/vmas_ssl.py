"""VmasSSL: VMAS-backed SSL simulator. Mirrors the RSimSSL interface."""

import math
from typing import List

import torch
from vmas import make_env

from utama_core.rsoccer_simulator.src.Entities import (
    Ball,
    Field,
    Frame,
    FrameSSL,
    Robot,
)
from utama_core.vmas_simulator.src.Scenario.ssl_scenario import SSLScenario
from utama_core.vmas_simulator.src.Utils.config import SSLFieldConfig, SSLScenarioConfig


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

    def reset(self, frame: Frame):
        """Reset using a Frame object (matches RSimSSL.reset interface)."""
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
        We directly set agent velocities rather than using force-based actions,
        since the PID controller already computes target velocities.
        """
        # all_agents = self.scenario.blue_agents + self.scenario.yellow_agents

        # Map commands by (yellow, id) for quick lookup
        cmd_map = {}
        for cmd in commands:
            cmd_map[(cmd.yellow, cmd.id)] = cmd

        # Apply velocity commands directly to agents
        for cmd in commands:
            if cmd.yellow:
                agent_idx = cmd.id
                if agent_idx >= self.n_robots_yellow:
                    continue
                agent = self.scenario.yellow_agents[agent_idx]
            else:
                agent_idx = cmd.id
                if agent_idx >= self.n_robots_blue:
                    continue
                agent = self.scenario.blue_agents[agent_idx]

            # Convert local velocity commands to global frame
            rot = agent.state.rot[0, 0].item()
            cos_r = math.cos(rot)
            sin_r = math.sin(rot)
            local_vx = float(cmd.v_x) if cmd.v_x else 0.0
            local_vy = float(cmd.v_y) if cmd.v_y else 0.0
            global_vx = local_vx * cos_r - local_vy * sin_r
            global_vy = local_vx * sin_r + local_vy * cos_r

            vel = torch.tensor([[global_vx, global_vy]], device=self.device, dtype=torch.float32).expand(
                self.num_envs, -1
            )
            agent.set_vel(vel, batch_index=None)

            # Angular velocity
            v_theta = float(cmd.v_theta) if cmd.v_theta else 0.0
            if hasattr(agent.state, "ang_vel") and agent.state.ang_vel is not None:
                ang_vel = torch.tensor([[v_theta]], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
                agent.state.ang_vel = ang_vel

            # Update rotation by integrating angular velocity
            dt = self.world.dt
            new_rot = agent.state.rot + v_theta * dt
            agent.set_rot(new_rot, batch_index=None)

        # Step the world physics (handles collisions)
        self.world.step()

        # Process kick/dribble actions
        for cmd in commands:
            if cmd.yellow:
                if cmd.id >= self.n_robots_yellow:
                    continue
                agent = self.scenario.yellow_agents[cmd.id]
            else:
                if cmd.id >= self.n_robots_blue:
                    continue
                agent = self.scenario.blue_agents[cmd.id]

            # Set up action tensor for process_action (kick/dribble)
            kick_val = float(cmd.kick_v_x) if cmd.kick_v_x else 0.0
            dribble_val = 1.0 if cmd.dribbler else 0.0
            action_u = torch.zeros(self.num_envs, 5, device=self.device, dtype=torch.float32)
            action_u[:, 3] = kick_val
            action_u[:, 4] = dribble_val
            agent.action.u = action_u
            self.scenario.process_action(agent)

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
        dc = self.scenario.cfg.dynamics
        for i, agent in enumerate(self.scenario.blue_agents):
            pos = agent.state.pos[0]
            vel = agent.state.vel[0]
            rot = agent.state.rot[0]

            # Determine infrared (ball contact) from proximity
            dist_to_ball = torch.norm(pos - ball_pos).item()
            has_ball = dist_to_ball < dc.dribble_dist_threshold

            robot = Robot(
                yellow=False,
                id=i,
                x=pos[0].item(),
                y=pos[1].item(),
                theta=math.degrees(rot[0].item()),
                v_x=vel[0].item(),
                v_y=vel[1].item(),
                v_theta=0.0,  # VMAS doesn't track angular velocity in the same way
                infrared=has_ball,
            )
            frame.robots_blue[i] = robot

        # Yellow robots
        for i, agent in enumerate(self.scenario.yellow_agents):
            pos = agent.state.pos[0]
            vel = agent.state.vel[0]
            rot = agent.state.rot[0]

            dist_to_ball = torch.norm(pos - ball_pos).item()
            has_ball = dist_to_ball < dc.dribble_dist_threshold

            robot = Robot(
                yellow=True,
                id=i,
                x=pos[0].item(),
                y=pos[1].item(),
                theta=math.degrees(rot[0].item()),
                v_x=vel[0].item(),
                v_y=vel[1].item(),
                v_theta=0.0,
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
