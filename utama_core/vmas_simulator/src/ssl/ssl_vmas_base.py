"""SSLVmasBaseEnv: Wraps VmasSSL like SSLBaseEnv wraps RSimSSL.

Provides the same step/reset/teleport interface used by the existing
controller layer. Uses SSL standard coordinate system (no Y-inversion).
"""

import math
from typing import List

import numpy as np
import pygame
import torch

from utama_core.global_utils.math_utils import rad_to_deg
from utama_core.rsoccer_simulator.src.Entities import Ball, Field, Frame, Robot
from utama_core.rsoccer_simulator.src.Render import (
    COLORS,
    RenderBall,
    RenderSSLRobot,
    SSLRenderField,
)
from utama_core.rsoccer_simulator.src.Render.overlay import (
    OverlayObject,
    OverlayType,
    RenderOverlay,
)
from utama_core.vmas_simulator.src.Simulators.vmas_ssl import VmasSSL
from utama_core.vmas_simulator.src.Utils.config import SSLScenarioConfig


class SSLVmasBaseEnv:
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
        "render.fps": 60,
    }
    NORM_BOUNDS = 1.2

    def __init__(
        self,
        field_type: int = 1,
        n_robots_blue: int = 6,
        n_robots_yellow: int = 6,
        time_step: float = 1.0 / 60.0,
        render_mode=None,
        num_envs: int = 1,
        device: str = "cpu",
        scenario_config: SSLScenarioConfig = None,
    ):
        self.render_mode = render_mode
        self.time_step = time_step
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow

        if scenario_config is None:
            scenario_config = SSLScenarioConfig(
                n_blue=n_robots_blue,
                n_yellow=n_robots_yellow,
            )

        self.vmas = VmasSSL(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=int(time_step * 1000),
            num_envs=num_envs,
            device=device,
            scenario_config=scenario_config,
        )

        self.field_type = field_type
        self.field: Field = self.vmas.get_field_params()
        self.max_pos = max(self.field.width / 2, (self.field.length / 2) + self.field.penalty_length)
        max_wheel_rad_s = (self.field.rbt_motor_max_rpm / 60) * 2 * np.pi
        self.max_v = max_wheel_rad_s * self.field.rbt_wheel_radius
        self.max_w = np.rad2deg(self.max_v / 0.095)

        self.frame: Frame = None
        self.last_frame: Frame = None
        self.steps = 0
        self.sent_commands = None

        # Render state (Pygame, matching RSim style)
        self.overlay: list[OverlayObject] = []
        self.field_renderer = SSLRenderField()
        self.window_surface = None
        self.window_size = self.field_renderer.window_size
        self.clock = None

    def step(self, action):
        self.steps += 1
        commands: List[Robot] = self._get_commands(action)
        self.vmas.send_commands(commands)
        self.sent_commands = commands

        self.last_frame = self.frame
        self.frame = self.vmas.get_frame()

        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()
        if self.render_mode == "human":
            self.render()
        return observation, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        initial_pos_frame: Frame = self._get_initial_positions_frame()
        self.vmas.reset(initial_pos_frame)

        self.frame = self.vmas.get_frame()
        obs = self._frame_to_observations()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def render(self):
        """Render the game using Pygame with RSim-style field and entities."""
        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("SSL VMAS Environment")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert self.window_surface is not None

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._render()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)),
                axes=(1, 0, 2),
            )

    def _render(self):
        self.field_renderer.draw(self.window_surface)

        if self.overlay:
            overlay = RenderOverlay(self.overlay, self.field_renderer.scale)
            overlay.draw(self.window_surface)

        for robot in self.frame.robots_blue.values():
            x, y = self._pos_transform(robot.x, robot.y)
            rbt = RenderSSLRobot(
                x,
                y,
                -robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["BLUE"],
            )
            rbt.draw(self.window_surface)

        for robot in self.frame.robots_yellow.values():
            x, y = self._pos_transform(robot.x, robot.y)
            rbt = RenderSSLRobot(
                x,
                y,
                -robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["YELLOW"],
            )
            rbt.draw(self.window_surface)

        ball = RenderBall(
            *self._pos_transform(self.frame.ball.x, self.frame.ball.y),
            self.field_renderer.scale,
        )
        ball.draw(self.window_surface)

        self.overlay = []

    def _pos_transform(self, pos_x, pos_y):
        return (
            int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
            int(-pos_y * self.field_renderer.scale + self.field_renderer.center_y),
        )

    def draw_point(self, x: float, y: float, color: str = "RED", width: float = 1):
        width = width if width >= 1 else 1
        point_data = OverlayObject(
            type=OverlayType.POINT,
            color=color,
            points=[self._pos_transform(x, y)],
            width=width,
        )
        self.overlay.append(point_data)

    def draw_line(self, points: list[tuple[float, float]], color: str = "RED", width: float = 1):
        width = width if width >= 1 else 1
        transformed_points = [self._pos_transform(p[0], p[1]) for p in points]
        line_data = OverlayObject(
            type=OverlayType.LINE,
            color=color,
            points=transformed_points,
            width=width,
        )
        self.overlay.append(line_data)

    def draw_polygon(self, points: list[tuple[float, float]], color: str = "RED", width: float = 1):
        width = width if width >= 1 else 1
        transformed_points = [self._pos_transform(p[0], p[1]) for p in points]
        poly_data = OverlayObject(
            type=OverlayType.POLYGON,
            color=color,
            points=transformed_points,
            width=width,
        )
        self.overlay.append(poly_data)

    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.window_surface = None
        self.vmas.stop()

    # --- Teleport methods (same interface as SSLBaseEnv) ---

    def teleport_ball(self, x: float, y: float, vx: float = 0, vy: float = 0):
        """Teleport ball to new position in meters.

        Uses SSL standard coordinates (no Y-inversion, unlike RSim).
        """
        ball = Ball(x=x, y=y, z=0.0, v_x=vx, v_y=vy)
        self.frame.ball = ball
        self.vmas.reset(self.frame)

    def teleport_robot(
        self,
        is_team_yellow: bool,
        robot_id: int,
        x: float,
        y: float,
        theta: float = None,
    ):
        """Teleport robot to new position in meters, radians.

        Uses SSL standard coordinates: theta is radians CCW from X-axis.
        """
        if theta is None:
            if is_team_yellow:
                theta = self.frame.robots_yellow[robot_id].theta
            else:
                theta = self.frame.robots_blue[robot_id].theta
        else:
            # theta comes in as radians from the strategy layer,
            # Frame stores degrees internally
            theta = rad_to_deg(theta)

        robot = Robot(yellow=is_team_yellow, id=robot_id, x=x, y=y, theta=theta)
        if is_team_yellow:
            self.frame.robots_yellow[robot_id] = robot
        else:
            self.frame.robots_blue[robot_id] = robot

        self.vmas.reset(self.frame)

    # --- Abstract methods (to be implemented by subclasses) ---

    def _get_commands(self, action):
        raise NotImplementedError

    def _frame_to_observations(self):
        raise NotImplementedError

    def _calculate_reward_and_done(self):
        raise NotImplementedError

    def _get_initial_positions_frame(self) -> Frame:
        raise NotImplementedError

    # --- Normalization utilities ---

    def norm_pos(self, pos):
        return np.clip(pos / self.max_pos, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def norm_v(self, v):
        return np.clip(v / self.max_v, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def norm_w(self, w):
        return np.clip(w / self.max_w, -self.NORM_BOUNDS, self.NORM_BOUNDS)
