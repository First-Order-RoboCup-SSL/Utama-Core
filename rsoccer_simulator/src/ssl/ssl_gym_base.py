"""
#   Environment Communication Structure
#    - Father class that creates the structure to communicate with multples setups of enviroment
#    - To create your wrapper from env to communcation, use inherit from this class! 
"""

import time
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pygame

from rsoccer_simulator.src.Entities import Field, Frame, Robot, Ball
from rsoccer_simulator.src.Render import (
    COLORS,
    RenderBall,
    SSLRenderField,
    RenderSSLRobot,
)
from rsoccer_simulator.src.Render.overlay import (
    RenderOverlay,
    OverlayObject,
    OverlayType,
)
from rsoccer_simulator.src.Simulators.rsim import RSimSSL

from global_utils.math_utils import rad_to_deg


class SSLBaseEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
        "render.fps": 60,
    }
    NORM_BOUNDS = 1.2

    def __init__(
        self,
        field_type: int,
        n_robots_blue: int,
        n_robots_yellow: int,
        time_step: float,
        render_mode=None,
    ):
        super().__init__()
        # Initialize Simulator
        self.render_mode = render_mode
        self.time_step = time_step
        self.rsim = RSimSSL(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=int(self.time_step * 1000),
        )
        self.n_robots_blue: int = n_robots_blue
        self.n_robots_yellow: int = n_robots_yellow

        # Get field dimensions
        self.field_type: int = field_type
        self.field: Field = self.rsim.get_field_params()
        self.max_pos = max(
            self.field.width / 2, (self.field.length / 2) + self.field.penalty_length
        )
        max_wheel_rad_s = (self.field.rbt_motor_max_rpm / 60) * 2 * np.pi
        self.max_v = max_wheel_rad_s * self.field.rbt_wheel_radius
        # 0.04 = robot radius (0.09) + wheel thicknees (0.005)
        self.max_w = np.rad2deg(self.max_v / 0.095)

        # Initiate
        self.frame: Frame = None
        self.last_frame: Frame = None
        self.steps = 0
        self.sent_commands = None
        self.overlay: list[OverlayObject] = []

        # Render
        self.field_renderer = SSLRenderField()
        self.window_surface = None
        self.window_size = self.field_renderer.window_size
        self.clock = None

    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands: List[Robot] = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()
        if self.render_mode == "human":
            self.render()
        return observation, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        initial_pos_frame: Frame = self._get_initial_positions_frame()
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()
        obs = self._frame_to_observations()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def render(self) -> None:
        """
        Renders the game depending on
        ball's and players' positions.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("SSL Environment")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        self._render()
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
        self.rsim.stop()

    ### CUSTOM FUNCTIONS WE ADDED ###

    def teleport_ball(self, x: float, y: float, vx: float = 0, vy: float = 0):
        """
        teleport ball to new position in meters

        Note: this does not create a new frame, but mutates the current frame
        """
        print(self.frame.ball)
        ball = Ball(x=x, y=y, z=self.frame.ball.z, v_x=vx, v_y=vy)
        self.frame.ball = ball
        self.rsim.reset(self.frame)

    def teleport_robot(
        self,
        is_team_yellow: bool,
        robot_id: bool,
        x: float,
        y: float,
        theta: float = None,
    ):
        """
        teleport robot to new position in meters, radians

        Note: this does not create a new frame, but mutates the current frame
        """
        if theta is None:
            if is_team_yellow:
                theta = self.frame.robots_yellow[robot_id].theta
            else:
                theta = self.frame.robots_blue[robot_id].theta
        else:
            theta = rad_to_deg(theta)

        robot = Robot(yellow=is_team_yellow, id=robot_id, x=x, y=y, theta=theta)
        if is_team_yellow:
            self.frame.robots_yellow[robot_id] = robot
        else:
            self.frame.robots_blue[robot_id] = robot

        self.rsim.reset(self.frame)

    def draw_point(self, x: float, y: float, color: str = "RED", width: float = 0.05):
        """
        draw a point on the field for debugging purposes. Rendered on next step() call.
        """
        point_data = OverlayObject(
            type=OverlayType.POINT,
            color=color,
            points=[self._pos_transform(x, y)],
            width=width,
        )
        self.overlay.append(point_data)

    def draw_line(
        self, points: list[tuple[float, float]], color: str = "RED", width: float = 0.05
    ):
        """
        draw a line on the field for debugging purposes. Rendered on next step() call.
            Note:   Only draws lines between the first and last point in the list.
                    If you want to draw multiple lines, call the draw_polygon method instead.
        """
        transformed_points = []
        for point in points:
            transformed_points.append(self._pos_transform(*point))
        line_data = OverlayObject(
            type=OverlayType.LINE, color=color, points=transformed_points, width=width
        )
        self.overlay.append(line_data)

    def draw_polygon(
        self, points: list[tuple[float, float]], color: str = "RED", width: float = 0.05
    ):
        """
        draw a polygon on the field for debugging purposes. Rendered on next step() call.
        """
        transformed_points = []
        for point in points:
            transformed_points.append(self._pos_transform(*point))
        poly_data = OverlayObject(
            type=OverlayType.POLYGON,
            color=color,
            points=transformed_points,
            width=width,
        )
        self.overlay.append(poly_data)

    ### END OF CUSTOM FUNCTIONS ###

    def _render(self):
        ball = RenderBall(
            *self._pos_transform(self.frame.ball.x, self.frame.ball.y),
            self.field_renderer.scale,
        )

        self.field_renderer.draw(self.window_surface)

        # added this for drawing overlays
        if self.overlay:
            overlay = RenderOverlay(self.overlay, self.field_renderer.scale)
            overlay.draw(self.window_surface)

        for i in range(self.n_robots_blue):
            robot = self.frame.robots_blue[i]
            x, y = self._pos_transform(robot.x, robot.y)
            rbt = RenderSSLRobot(
                x,
                y,
                robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["BLUE"],
            )
            rbt.draw(self.window_surface)

        for i in range(self.n_robots_yellow):
            robot = self.frame.robots_yellow[i]
            x, y = self._pos_transform(robot.x, robot.y)
            rbt = RenderSSLRobot(
                x,
                y,
                robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["YELLOW"],
            )
            rbt.draw(self.window_surface)
        ball.draw(self.window_surface)

        self.overlay = []  # clear overlay after render

    def _pos_transform(self, pos_x, pos_y):
        return (
            int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
            int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
        )

    def _get_commands(self, action):
        """returns a list of commands of type List[Robot] from type action_space action"""
        raise NotImplementedError

    def _frame_to_observations(self):
        """returns a type observation_space observation from a type List[Robot] state"""
        raise NotImplementedError

    def _calculate_reward_and_done(self):
        """returns reward value and done flag from type List[Robot] state"""
        raise NotImplementedError

    def _get_initial_positions_frame(self) -> Frame:
        """returns frame with robots initial positions"""
        raise NotImplementedError

    def norm_pos(self, pos):
        return np.clip(pos / self.max_pos, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def norm_v(self, v):
        return np.clip(v / self.max_v, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def norm_w(self, w):
        return np.clip(w / self.max_w, -self.NORM_BOUNDS, self.NORM_BOUNDS)
