"""VmasRobotController: Mirrors RSimRobotController for the VMAS backend.

Uses SSL standard coordinate system — commands are passed through directly
(like GRSimRobotController), with no Y-axis or angular velocity inversions.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from utama_core.entities.data.command import RobotCommand, RobotResponse
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from utama_core.vmas_simulator.src.ssl.ssl_vmas_base import SSLVmasBaseEnv

logger = logging.getLogger(__name__)


class VmasRobotController(AbstractRobotController):
    """Robot Controller (and Vision Receiver) for VMAS.

    Same interface as RSimRobotController. Commands use SSL standard frame
    — direct passthrough like GRSim (no coordinate inversions needed).
    """

    def __init__(
        self,
        is_team_yellow: bool,
        n_friendly: int,
        env: SSLVmasBaseEnv,
        pvp_manager=None,
    ):
        super().__init__(is_team_yellow, n_friendly)
        self._env = env
        self._n_enemy = self._get_n_enemy()
        self._out_packet = self._empty_command(self._n_friendly)
        self._pvp_manager = pvp_manager

        if not self.pvp_manager:
            self.env.reset()

    def get_robots_responses(self) -> Optional[List[RobotResponse]]:
        return self._robots_info.popleft() if len(self._robots_info) > 0 else None

    def send_robot_commands(self) -> None:
        if self.pvp_manager:
            self.pvp_manager.send_command(self.is_team_yellow, self._out_packet)
        else:
            if self.is_team_yellow:
                action = {
                    "team_blue": tuple(self._empty_command(self.n_enemy)),
                    "team_yellow": tuple(self._out_packet),
                }
            else:
                action = {
                    "team_blue": tuple(self._out_packet),
                    "team_yellow": tuple(self._empty_command(self.n_enemy)),
                }

            observation, reward, terminated, truncated, reward_shaping = self._env.step(action)

            new_frame, yellow_robots_info, blue_robots_info = observation
            if self.is_team_yellow:
                self._robots_info.append(yellow_robots_info)
            else:
                self._robots_info.append(blue_robots_info)

            logger.debug(f"{new_frame} {terminated} {truncated} {reward_shaping}")
            self._out_packet = self._empty_command(self.n_friendly)

    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        super().add_robot_commands(robot_commands, robot_id)

    def update_robots_info(self, robots_info):
        self._robots_info.append(robots_info)

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """Add a robot command. SSL standard frame — direct passthrough (no inversions)."""
        action = np.array(
            [
                command.local_forward_vel,
                command.local_left_vel,  # No negation (unlike RSim)
                command.angular_vel,  # No negation (unlike RSim)
                command.kick,
                command.dribble,
            ],
            dtype=np.float32,
        )
        self._out_packet[robot_id] = action

    def empty_command(self) -> list[NDArray]:
        return self._empty_command(self.n_friendly)

    def _empty_command(self, n_robots: int) -> list[NDArray]:
        return [np.zeros((6,), dtype=float) for _ in range(n_robots)]

    def _get_n_enemy(self) -> int:
        n_yellow = self._env.n_robots_yellow
        n_blue = self._env.n_robots_blue
        if self._is_team_yellow:
            assert n_yellow == self._n_friendly
            return n_blue
        else:
            assert n_blue == self._n_friendly
            return n_yellow

    @property
    def is_team_yellow(self):
        return self._is_team_yellow

    @property
    def env(self):
        return self._env

    @property
    def robots_info(self):
        return self._robots_info

    @property
    def pvp_manager(self):
        return self._pvp_manager

    @property
    def n_friendly(self):
        return self._n_friendly

    @property
    def n_enemy(self):
        return self._n_enemy


class VmasPVPManager:
    """Manages player vs player in the VMAS environment.

    Mirrors RSimPVPManager: synchronizes two controllers on a single environment.
    """

    def __init__(self, env: SSLVmasBaseEnv):
        self._env = env
        self.n_robots_blue = env.n_robots_blue
        self.n_robots_yellow = env.n_robots_yellow
        self._pending = {"team_blue": None, "team_yellow": None}
        self.blue_player = None
        self.yellow_player = None
        self._env.reset()

    def load_controllers(
        self,
        yellow_controller: VmasRobotController,
        blue_controller: VmasRobotController,
    ):
        self.blue_player = blue_controller
        self.yellow_player = yellow_controller

    def send_command(self, is_yellow: bool, out_packet: list[NDArray]):
        assert (
            self.blue_player is not None and self.yellow_player is not None
        ), "Blue and yellow players must be set before sending commands."

        colour = "team_yellow" if is_yellow else "team_blue"
        other_colour = "team_blue" if is_yellow else "team_yellow"

        self._pending[colour] = tuple(out_packet)
        if self._pending[other_colour]:
            observation, reward, terminated, truncated, reward_shaping = self._env.step(self._pending)

            new_frame, yellow_robots_info, blue_robots_info = observation
            self.blue_player.update_robots_info(blue_robots_info)
            self.yellow_player.update_robots_info(yellow_robots_info)

            self._pending = {"team_blue": None, "team_yellow": None}

    def _empty_command(self, n_robots: int) -> list[NDArray]:
        return [np.zeros((6,), dtype=float) for _ in range(n_robots)]
