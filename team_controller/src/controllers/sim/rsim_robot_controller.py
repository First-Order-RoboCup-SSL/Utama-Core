from curses.ascii import RS
from typing import Dict, Union, Optional, Tuple
from xmlrpc.client import Boolean
from entities.game import Game
from entities.data.command import RobotCommand, RobotInfo
from entities.data.vision import FrameData
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
import numpy as np
from numpy.typing import NDArray
from entities.data.command import RobotCommand, RobotInfo
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv


class PVPManager:
    def __init__(self, env: SSLBaseEnv, n_robots: int):
        self._env = env
        self.n_robots = n_robots
        self._pending = {
            "team_blue": None,
            "team_yellow": None
        }

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
                self._pending[colour] = tuple(self._empty_command(self.n_enemy_robots))
        
        observation, reward, terminated, truncated, reward_shaping = self._env.step(self._pending[colour])


        # FIXXX HERE 
        # note that we should not technically be able to view the opponent's robots_info!!
        new_frame, yellow_robots_info, blue_robots_info = observation
        if self.is_team_yellow:
            self._robots_info = yellow_robots_info
        else:
            self._robots_info = blue_robots_info

        if self._debug:
            print(new_frame, terminated, truncated, reward_shaping)

        self._write_to_game_obj(new_frame)
        # flush out_packet
        self._out_packet = self._empty_command(self.n_friendly_robots)


        self._pending = {
            "team_blue": None,
            "team_yellow": None
        }

class RSimRobotController(AbstractRobotController):
    """
    Robot Controller (and Vision Receiver) for RSim.

    is_pvp:
    if pvp, two controllers are playing against each other. Else, play against static

    There is no need for a separate Vision Receiver for RSim.
    """

    def __init__(
        self,
        is_team_yellow: bool,
        env: SSLBaseEnv,
        pvp_manager: Optional[PVPManager],
        game_obj: Game,
        debug: bool = False,
    ):
        self._is_team_yellow = is_team_yellow
        self._game_obj = game_obj
        self._debug = debug
        self._env = env
        self._n_friendly_robots, self._n_enemy_robots = self._get_n_robots()
        self._out_packet = self._empty_command(self.n_friendly_robots)
        self._robots_info: list[RobotInfo] = [None] * self.n_friendly_robots
        self.is_pvp = pvp_manager is not None
        self.pvp_manager = pvp_manager

        # if environment was not reset beforehand, reset now
        if self._env.frame is None:
            initial_obs, _ = self._env.reset()
            initial_frame = initial_obs[0]
        else:
            initial_frame, _, _ = self._env._frame_to_observations()
        self._write_to_game_obj(initial_frame)

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

            if self._debug:
                print(new_frame, terminated, truncated, reward_shaping)

            self._write_to_game_obj(new_frame)
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

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """
        Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick_spd', 'kick_angle', 'dribbler_spd'.
        """
        # invert angular_vel and left_vel because axis is inverted in RSim
        action = np.array(
            [
                command.local_forward_vel,
                -command.local_left_vel,
                command.angular_vel,
                command.kick_spd,
                command.dribbler_spd,
            ],
            dtype=np.float32,
        )
        self._out_packet[robot_id] = action

    def _write_to_game_obj(self, new_frame: FrameData) -> None:
        """
        Supersedes the VisionReceiver and queue procedure to write to game obj directly.

        Done this way, because there's no separate vision receivere for RSim.
        """
        self._game_obj.add_new_state(new_frame)

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
            if self.debug:
                print(f"Robot: {robot_id} has the Ball")
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
    def env(self):
        return self._env

    @property
    def game_obj(self):
        return self._game_obj

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
