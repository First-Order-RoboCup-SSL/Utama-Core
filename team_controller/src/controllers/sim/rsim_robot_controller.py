from typing import Dict, Union, Optional
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


class RSimRobotController(AbstractRobotController):
    """
    Robot Controller (and Vision Receiver) for RSim.

    There is no need for a separate Vision Receiver for RSim.
    """

    def __init__(
        self,
        is_team_yellow: bool,
        env: SSLBaseEnv,
        game_obj: Game,
        debug: bool = False,
    ):
        self._is_team_yellow = is_team_yellow
        self._game_obj = game_obj
        self._debug = debug
        self._env = env
        self._out_packet = self._empty_command()
        self._robots_info: list[RobotInfo] = [None] * 6

        initial_obs, _ = self._env.reset()
        initial_frame = initial_obs[0]
        self._write_to_game_obj(initial_frame)

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).
        """
        action = {
            "team_blue": tuple(self._empty_command()),
            "team_yellow": tuple(self._out_packet),
        }
        # print(action)
        observation, reward, terminated, truncated, reward_shaping = self._env.step(
            action
        )

        # note that we should not technically be able to view the opponent's robots_info!!
        new_frame, yellow_robots_info, blue_robots_info = observation

        if self._debug:
            print(new_frame, terminated, truncated, reward_shaping)

        self._write_to_game_obj(new_frame)
        # flush out_packet
        self._out_packet = self._empty_command()

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
        action = np.array(
            [
                command.local_forward_vel,
                command.local_left_vel,
                command.angular_vel,
                command.kick_spd,
                command.kick_angle,
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
    def _empty_command(self) -> list[NDArray]:
        return [np.zeros((6,), dtype=float) for _ in range(6)]

    def robot_has_ball(self, robot_id: int) -> bool:
        """
        Checks if the specified robot has the ball.

        Args:
            robot_id (int): The ID of the robot.

        Returns:
            bool: True if the robot has the ball, False otherwise.
        """
        for id, robot_feedback in enumerate(self.robots_info):
            if robot_feedback != None:
                if robot_feedback.has_ball and id == robot_id:
                    if self.debug:
                        print(f"Robot: {robot_id}: HAS the Ball")
                    return True
                else:
                    return False

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
