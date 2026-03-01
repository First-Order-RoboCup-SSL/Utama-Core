import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from utama_core.entities.data.command import RobotCommand, RobotResponse
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

logger = logging.getLogger(__name__)


class RSimRobotController(AbstractRobotController):
    """Robot Controller (and Vision Receiver) for RSim.

    Note: also does the first reset for the environment.

    pvp_manager:
    if not None, two controllers are playing against each other. Else, play against static

    There is no need for a separate Vision Receiver for RSim.
    """

    def __init__(
        self,
        is_team_yellow: bool,
        n_friendly: int,
        env: SSLBaseEnv,
        pvp_manager: Optional["RSimPVPManager"] = None,
    ):
        super().__init__(is_team_yellow, n_friendly)
        self._env = env
        self._n_enemy = self._get_n_enemy()
        self._out_packet = self._empty_command(self._n_friendly)
        self._pvp_manager = pvp_manager

        if self._pvp_manager is None:
            self._env.reset()

    def get_robots_responses(self) -> Optional[List[RobotResponse]]:
        return self._robots_info.popleft() if len(self._robots_info) > 0 else None

    def send_robot_commands(self) -> None:
        """Sends the robot commands to the appropriate team (yellow or blue)."""
        if self._pvp_manager is not None:
            self._pvp_manager.send_command(self.is_team_yellow, self._out_packet)
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

            # note that we should not technically be able to view the opponent's robots_info!!
            new_frame, yellow_robots_info, blue_robots_info = observation
            if self.is_team_yellow:
                self._robots_info.append(yellow_robots_info)
            else:
                self._robots_info.append(blue_robots_info)

            logger.debug(f"{new_frame} {terminated} {truncated} {reward_shaping}")

            # flush out_packet
            self._out_packet = self._empty_command(self.n_friendly)

    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """Adds robot commands to the out_packet.

        Args:
            robot_commands (Union[RobotCommand, Dict[int, RobotCommand]]): A single RobotCommand or a dictionary of RobotCommand with robot_id as the key.
            robot_id (Optional[int]): The ID of the robot which is ONLY used when adding one Robot command. Defaults to None.

        Raises:
            SyntaxWarning: If invalid hyperparameters are passed to the function.

        Calls add_robot_command for each entered command
        """
        super().add_robot_commands(robot_commands, robot_id)

    def update_robots_info(self, robots_info):
        """Updates robots info to input.

        Used by PVPManager to update robots info.
        """
        self._robots_info.append(robots_info)

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick', 'chip', 'dribble'.
        """
        # invert angular_vel and left_vel because axis is inverted in RSim
        action = np.array(
            [
                command.local_forward_vel,
                -command.local_left_vel,
                -command.angular_vel,
                command.kick,
                command.dribble,
            ],
            dtype=np.float32,
        )
        self._out_packet[robot_id] = action

    def empty_command(self) -> list[NDArray]:
        return self._empty_command(self.n_friendly)

    # create an empty command array
    def _empty_command(self, n_robots: int) -> list[NDArray]:
        return [np.zeros((6,), dtype=float) for _ in range(n_robots)]

    def _get_n_enemy(self) -> Tuple[int, int]:
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


class RSimPVPManager:
    """Manages a player vs player game inside the rsim environment.

    The two teams run in lockstep in this setup, and so, in order to get results consistent with running just one
    player, it's important to either alternate the player colours that send commands from the main loop (using an empty
    command if one team has nothing to do), or call flush() after every command that should be processed on its own
    (without a corresponding command from the other team).
    """

    def __init__(self, env: SSLBaseEnv):
        self._env = env
        self.n_robots_blue = env.n_robots_blue
        self.n_robots_yellow = env.n_robots_yellow
        self._pending = {"team_blue": None, "team_yellow": None}
        self.blue_player = None
        self.yellow_player = None
        self._env.reset()

    # TODO: this is clumsy af and it can be removed entirely once we remove robot_has_ball function from robot_controller
    def load_controllers(
        self,
        yellow_controller: RSimRobotController,
        blue_controller: RSimRobotController,
    ):
        """Loads the blue and yellow controllers."""
        self.blue_player = blue_controller
        self.yellow_player = yellow_controller

    def send_command(self, is_yellow: bool, out_packet: list[NDArray]):
        """Sends the robot commands to the appropriate team (yellow or blue)."""
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
