from typing import Dict, Union, Optional, Tuple
from xmlrpc.client import Boolean
from entities.data.raw_vision import RawVisionData
from entities.game import Game
from entities.data.command import RobotCommand, RobotResponse
from refiners.position import PositionRefiner
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
import numpy as np
from numpy.typing import NDArray
from entities.data.command import RobotCommand, RobotResponse
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
import logging

logger = logging.getLogger(__name__)


class RSimRobotController(AbstractRobotController):
    """
    Robot Controller (and Vision Receiver) for RSim.

    pvp_manager:
    if not None, two controllers are playing against each other. Else, play against static

    There is no need for a separate Vision Receiver for RSim.
    """

    def __init__(
        self,
        is_team_yellow: bool,
        env: SSLBaseEnv,
        is_pvp,
    ):
        self._is_team_yellow = is_team_yellow
        self._last_frame = None
        self._env = env
        self._n_friendly_robots, self._n_enemy_robots = self._get_n_robots()
        self._out_packet = self._empty_command(self.n_friendly_robots)
        self._robots_info: list[RobotResponse] = [None] * self.n_friendly_robots
        self._is_pvp = is_pvp

        if not self._is_pvp:
            self.reset_env()

    def reset_env(self):
        # if environment was not reset beforehand, reset now
        if self._env.frame is None:
            initial_obs, _ = self._env.reset()
            initial_frame = initial_obs[0]
        else:
            initial_frame, _, _ = self._env._frame_to_observations()

        self._last_frame = initial_frame

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).
        """
        if self.is_pvp:
            self.pvp_manager.send_command(self.is_team_yellow, self._out_packet)
        else:
            if self.is_team_yellow:
                action = {
                    "team_blue": tuple(self._empty_command(self.n_enemy_robots)),
                    "team_yellow": tuple(self._out_packet),
                }
            else:
                action = {
                    "team_blue": tuple(self._out_packet),
                    "team_yellow": tuple(self._empty_command(self.n_enemy_robots)),
                }

            observation, reward, terminated, truncated, reward_shaping = self._env.step(
                action
            )

            # note that we should not technically be able to view the opponent's robots_info!!
            new_frame, yellow_robots_info, blue_robots_info = observation
            if self.is_team_yellow:
                self._robots_info = yellow_robots_info
            else:
                self._robots_info = blue_robots_info

            logger.debug(f"{new_frame} {terminated} {truncated} {reward_shaping}")

            self._last_frame = new_frame
            # flush out_packet
            self._out_packet = self._empty_command(self.n_friendly_robots)

    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """
        Adds robot commands to the out_packet.

        Args:
            robot_commands (Union[RobotCommand, Dict[int, RobotCommand]]): A single RobotCommand or a dictionary of RobotCommand with robot_id as the key.
            robot_id (Optional[int]): The ID of the robot which is ONLY used when adding one Robot command. Defaults to None.

        Raises:
            SyntaxWarning: If invalid hyperparameters are passed to the function.

        Calls add_robot_command for each entered command
        """
        super().add_robot_commands(robot_commands, robot_id)

    def update_robots_info(self, robots_info):
        """
        Updates robots info to input. Used by PVPManager to update robots info.
        """
        self._robots_info = robots_info

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """
        Adds a robot command to the out_packet.

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
        return self._empty_command(self.n_friendly_robots)

    # create an empty command array
    def _empty_command(self, n_robots: int) -> list[NDArray]:
        return [np.zeros((6,), dtype=float) for _ in range(n_robots)]

    def robot_has_ball(self, robot_id: int) -> bool:
        """
        Checks if the specified robot has the ball.

        Args:
            robot_id (int): The ID of the robot.

        Returns:
            bool: True if the robot has the ball, False otherwise.
        """
        if self._robots_info[robot_id] is None:
            return False

        if self._robots_info[robot_id].has_ball:
            logger.debug(f"Robot: {robot_id} has the Ball")
            return True
        else:
            return False

    def _get_n_robots(self) -> Tuple[int, int]:
        n_yellow = self._env.n_robots_yellow
        n_blue = self._env.n_robots_blue
        if self._is_team_yellow:
            return n_yellow, n_blue
        else:
            return n_blue, n_yellow

    @property
    def is_team_yellow(self):
        return self._is_team_yellow

    @property
    def last_frame(self):
        return self._last_frame

    @property
    def env(self):
        return self._env

    @property
    def debug(self):
        return self._debug

    @property
    def robots_info(self):
        return self._robots_info

    @property
    def is_pvp(self):
        return self._is_pvp

    @property
    def n_friendly_robots(self):
        return self._n_friendly_robots

    @property
    def n_enemy_robots(self):
        return self._n_enemy_robots


class RSimPVPRobotController:
    """
    Manages a player vs player game inside the rsim environment. The two teams run in lockstep in
    this setup, and so, in order to get results consistent with running just one player,
    it's important to either alternate the player colours that send commands from the main loop
    (using an empty command if one team has nothing to do), or call flush() after every command that
    should be processed on its own (without a corresponding command from the other team).
    """

    def __init__(
        self,
        env: SSLBaseEnv,
        yellow_controller: RSimRobotController,
        blue_controller: RSimRobotController,
    ):
        self._env = env
        self.n_robots_blue = env.n_robots_blue
        self.n_robots_yellow = env.n_robots_yellow
        self._pending = {"team_blue": None, "team_yellow": None}
        self.last_frame = None
        self.blue_player = blue_controller
        self.yellow_player = yellow_controller
        self.reset_env()

    def send_command(self, is_yellow: Boolean, out_packet):
        colour = "team_yellow" if is_yellow else "team_blue"
        other_colour = "team_blue" if is_yellow else "team_yellow"

        if self._pending[colour]:
            self._fill_and_send()

        self._pending[colour] = tuple(out_packet)
        if self._pending[other_colour]:
            self._fill_and_send()

    def _empty_command(self, n_robots: int) -> list[NDArray]:
        return [np.zeros((6,), dtype=float) for _ in range(n_robots)]

    def _fill_and_send(self):
        for colour in ("team_blue", "team_yellow"):
            if not self._pending[colour]:
                self._pending[colour] = tuple(
                    self._empty_command(
                        self.n_robots_yellow
                        if colour == "team_yellow"
                        else self.n_robots_blue
                    )
                )

        observation, reward, terminated, truncated, reward_shaping = self._env.step(
            self._pending
        )

        new_frame, yellow_robots_info, blue_robots_info = observation
        self.blue_player.update_robots_info(blue_robots_info)
        self.yellow_player.update_robots_info(yellow_robots_info)

        self.last_frame = new_frame

        self._pending = {"team_blue": None, "team_yellow": None}

    def reset_env(self):
        # if environment was not reset beforehand, reset now
        if self._env.frame is None:
            initial_obs, _ = self._env.reset()
            initial_frame = initial_obs[0]
        else:
            initial_frame, _, _ = self._env._frame_to_observations()
        self.last_frame = initial_frame

    def flush(self):
        self._fill_and_send()
