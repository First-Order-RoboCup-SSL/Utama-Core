"""VmasStandardEnv: Mirrors SSLStandardEnv for the VMAS backend.

Outputs the same (RawVisionData, yellow_robots_info, blue_robots_info) tuple
as the rsim SSLStandardEnv, so StrategyRunner and the BT pipeline work unchanged.

Uses SSL standard coordinate system (like GRSim): no Y-axis or theta inversion.
"""

import logging
import random
from typing import List, Tuple

from numpy.random import normal

from utama_core.config.formations import LEFT_START_ONE, RIGHT_START_ONE
from utama_core.config.settings import (
    MAX_BALL_SPEED,
    MIN_RELEASE_SPEED,
    RELEASE_GAIN,
    TIMESTEP,
)
from utama_core.entities.data.command import RobotResponse
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.global_utils.math_utils import deg_to_rad, normalise_heading_deg
from utama_core.rsoccer_simulator.src.Entities import Ball, Frame, Robot
from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise
from utama_core.vmas_simulator.src.ssl.ssl_vmas_base import SSLVmasBaseEnv
from utama_core.vmas_simulator.src.Utils.config import SSLScenarioConfig

logger = logging.getLogger(__name__)


class VmasStandardEnv(SSLVmasBaseEnv):
    """Lightweight VMAS-backed SSL environment for testing, development, and RL training.

    Produces the same observation format as rsim's SSLStandardEnv so controllers
    and StrategyRunner can use this as a drop-in replacement.
    """

    def __init__(
        self,
        field_type: int = 1,
        render_mode: str = "human",
        n_robots_blue: int = 6,
        n_robots_yellow: int = 6,
        time_step: float = TIMESTEP,
        blue_starting_formation: list[tuple] = None,
        yellow_starting_formation: list[tuple] = None,
        gaussian_noise: RsimGaussianNoise = RsimGaussianNoise(),
        vanishing: float = 0,
        num_envs: int = 1,
        device: str = "cpu",
        scenario_config: SSLScenarioConfig = None,
    ):
        super().__init__(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step=time_step,
            render_mode=render_mode,
            num_envs=num_envs,
            device=device,
            scenario_config=scenario_config,
        )

        self.blue_formation = LEFT_START_ONE if not blue_starting_formation else blue_starting_formation
        self.yellow_formation = RIGHT_START_ONE if not yellow_starting_formation else yellow_starting_formation

        # Dribbler release tracking (same as rsim SSLStandardEnv)
        self.prev_dribbler_blue = [False] * self.n_robots_blue
        self.prev_dribbler_yellow = [False] * self.n_robots_yellow
        self.prev_forward_blue = [0.0] * self.n_robots_blue
        self.prev_forward_yellow = [0.0] * self.n_robots_yellow
        self.kick_persist_blue = [(0.0, 0) for _ in range(self.n_robots_blue)]
        self.kick_persist_yellow = [(0.0, 0) for _ in range(self.n_robots_yellow)]

        self.latest_observation = (-1, None)
        self.reward_shaping_total = None

        logger.info(f"{n_robots_blue}v{n_robots_yellow} VMAS SSL Environment Initialized")

        self.gaussian_noise = gaussian_noise
        assert vanishing >= 0, "Negative vanishing probability not allowed"
        self.vanishing = vanishing

    def reset(self, *, seed=None, options=None):
        self.reward_shaping_total = None
        for i in range(self.n_robots_blue):
            self.prev_dribbler_blue[i] = False
            self.prev_forward_blue[i] = 0.0
            self.kick_persist_blue[i] = (0.0, 0)
        for i in range(self.n_robots_yellow):
            self.prev_dribbler_yellow[i] = False
            self.prev_forward_yellow[i] = 0.0
            self.kick_persist_yellow[i] = (0.0, 0)
        self.latest_observation = (-1, None)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.steps += 1
        commands = self._get_commands(action)
        self._apply_dribbler_release_kicks(commands)

        self.vmas.send_commands(commands)
        self.sent_commands = commands

        self.last_frame = self.frame
        self.frame = self.vmas.get_frame()
        self._update_dribbler_history(commands)

        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()
        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, self.reward_shaping_total

    def _frame_to_observations(
        self,
    ) -> Tuple[RawVisionData, list[RobotResponse], list[RobotResponse]]:
        """Return observation matching the rsim SSLStandardEnv format.

        No Y-axis or theta inversion: VMAS uses SSL standard frame.
        """
        if self.latest_observation[0] == self.steps:
            return self.latest_observation[1]

        # Ball
        if self._vanishing():
            ball_obs = []
        else:
            self._add_gaussian_noise_ball(self.frame.ball, self.gaussian_noise)
            ball_obs = [RawBallData(self.frame.ball.x, self.frame.ball.y, self.frame.ball.z, 1.0)]

        # Blue robots
        blue_obs = []
        blue_robots_info = []
        for i in range(len(self.frame.robots_blue)):
            if self._vanishing():
                continue
            robot = self.frame.robots_blue[i]
            robot_pos, robot_info = self._get_robot_observation(robot)
            blue_obs.append(robot_pos)
            blue_robots_info.append(robot_info)

        # Yellow robots
        yellow_obs = []
        yellow_robots_info = []
        for i in range(len(self.frame.robots_yellow)):
            if self._vanishing():
                continue
            robot = self.frame.robots_yellow[i]
            robot_pos, robot_info = self._get_robot_observation(robot)
            yellow_obs.append(robot_pos)
            yellow_robots_info.append(robot_info)

        result = (
            RawVisionData(self.time_step * self.steps, yellow_obs, blue_obs, ball_obs, 0),
            yellow_robots_info,
            blue_robots_info,
        )
        self.latest_observation = (self.steps, result)
        return result

    def _get_robot_observation(self, robot):
        """Convert a Robot entity to RawRobotData + RobotResponse.

        SSL standard frame: no coordinate inversions needed.
        """
        self._add_gaussian_noise_robot(robot, self.gaussian_noise)

        robot_pos = RawRobotData(
            robot.id,
            robot.x,
            robot.y,
            float(deg_to_rad(robot.theta)),
            1,
        )
        robot_info = RobotResponse(robot.id, robot.infrared)
        return robot_pos, robot_info

    def _get_commands(self, actions) -> list[Robot]:
        commands = []

        for i in range(self.n_robots_blue):
            v_x = actions["team_blue"][i][0]
            v_y = actions["team_blue"][i][1]
            v_theta = actions["team_blue"][i][2]
            dribbler = actions["team_blue"][i][4] > 0
            kick_v_x = self.field.goal_depth if actions["team_blue"][i][3] > 0 else 0.0  # placeholder

            cmd = Robot(
                yellow=False,
                id=i,
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=kick_v_x,
                dribbler=dribbler,
            )
            commands.append(cmd)

        for i in range(self.n_robots_yellow):
            v_x = actions["team_yellow"][i][0]
            v_y = actions["team_yellow"][i][1]
            v_theta = actions["team_yellow"][i][2]
            dribbler = actions["team_yellow"][i][4] > 0
            kick_v_x = self.field.goal_depth if actions["team_yellow"][i][3] > 0 else 0.0

            cmd = Robot(
                yellow=True,
                id=i,
                v_x=v_x,
                v_y=v_y,
                v_theta=v_theta,
                kick_v_x=kick_v_x,
                dribbler=dribbler,
            )
            commands.append(cmd)

        return commands

    def _apply_dribbler_release_kicks(self, commands: list[Robot]) -> None:
        """Approximate kicks when dribblers turn off (same logic as rsim SSLStandardEnv)."""
        KICK_PERSISTENCE_FRAMES = 3

        n_blue = self.n_robots_blue
        for i in range(n_blue):
            spd, frames = self.kick_persist_blue[i]
            if frames > 0:
                self.kick_persist_blue[i] = (spd, frames - 1)
                commands[i].kick_v_x = max(commands[i].kick_v_x, spd)

            release = self._dribbler_release_kick(
                self.prev_dribbler_blue, self.prev_forward_blue, i, commands[i].dribbler
            )
            if release > 0.0:
                self.kick_persist_blue[i] = (release, KICK_PERSISTENCE_FRAMES)
                commands[i].kick_v_x = max(commands[i].kick_v_x, release)

        for j in range(self.n_robots_yellow):
            cmd_idx = n_blue + j
            spd, frames = self.kick_persist_yellow[j]
            if frames > 0:
                self.kick_persist_yellow[j] = (spd, frames - 1)
                commands[cmd_idx].kick_v_x = max(commands[cmd_idx].kick_v_x, spd)

            release = self._dribbler_release_kick(
                self.prev_dribbler_yellow, self.prev_forward_yellow, j, commands[cmd_idx].dribbler
            )
            if release > 0.0:
                self.kick_persist_yellow[j] = (release, KICK_PERSISTENCE_FRAMES)
                commands[cmd_idx].kick_v_x = max(commands[cmd_idx].kick_v_x, release)

    def _update_dribbler_history(self, commands: list[Robot]) -> None:
        n_blue = self.n_robots_blue
        for i in range(n_blue):
            self.prev_dribbler_blue[i] = commands[i].dribbler
            self.prev_forward_blue[i] = commands[i].v_x
        for j in range(self.n_robots_yellow):
            self.prev_dribbler_yellow[j] = commands[n_blue + j].dribbler
            self.prev_forward_yellow[j] = commands[n_blue + j].v_x

    def _dribbler_release_kick(
        self,
        prev_dribbler: List[bool],
        prev_forward: List[float],
        index: int,
        dribbler: bool,
    ) -> float:
        if not prev_dribbler[index] or dribbler:
            return 0.0
        forward = prev_forward[index]
        if forward < MIN_RELEASE_SPEED:
            return 0.0
        return min(RELEASE_GAIN * forward, MAX_BALL_SPEED)

    def _calculate_reward_and_done(self):
        return 1, False

    def _get_initial_positions_frame(self):
        """Returns initial frame. Uses SSL standard coordinates (no inversions)."""
        pos_frame = Frame()

        for i in range(self.n_robots_blue):
            x, y, heading = self.blue_formation[i]
            # formations.py uses radians for heading, Frame stores degrees
            pos_frame.robots_blue[i] = Robot(id=i, x=x, y=y, theta=float(heading * 180.0 / 3.141592653589793))

        for i in range(self.n_robots_yellow):
            x, y, heading = self.yellow_formation[i]
            pos_frame.robots_yellow[i] = Robot(id=i, x=x, y=y, theta=float(heading * 180.0 / 3.141592653589793))

        pos_frame.ball = Ball(x=0, y=0)
        return pos_frame

    def _vanishing(self) -> bool:
        return self.steps > 0 and self.vanishing and (random.random() < self.vanishing)

    @staticmethod
    def _add_gaussian_noise_ball(ball: Ball, noise: RsimGaussianNoise):
        if noise.x_stddev:
            ball.x += normal(scale=noise.x_stddev)
        if noise.y_stddev:
            ball.y += normal(scale=noise.y_stddev)

    @staticmethod
    def _add_gaussian_noise_robot(robot: Robot, noise: RsimGaussianNoise):
        if noise.x_stddev:
            robot.x += normal(scale=noise.x_stddev)
        if noise.y_stddev:
            robot.y += normal(scale=noise.y_stddev)
        if noise.th_stddev_deg:
            robot.theta = normalise_heading_deg(robot.theta + normal(scale=noise.th_stddev_deg))
