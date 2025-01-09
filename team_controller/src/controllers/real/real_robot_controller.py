from serial import Serial
from entities.game import Game

from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from team_controller.src.config.settings import BAUD_RATE, PORT, TIMEOUT
import logging

logger = logging.getLogger(__name__)

class RealRobotController(AbstractRobotController):
    def _init_(self, is_team_yellow: bool, game_obj: Game, debug: bool = False):
        self._is_team_yellow = is_team_yellow
        self._game_obj = game_obj
        self._debug = debug
        self._serial = Serial(port=PORT, baudrate=BAUD_RATE, timeout=TIMEOUT)

        logger.debug(
            f"Serial port: {PORT} opened with baudrate: {BAUD_RATE} and timeout {TIMEOUT}"
        )

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).
        """
        # TODO: add bytes to write for serial communication
        pass

    def add_robot_commands(self, robot_id: int, velocity: float, angle: float) -> None:
        """
        Adds robot commands to the packet to be sent to the robot.
        """
        # TODO: add robot commands to the packet
        pass

    @property
    def is_team_yellow(self) -> bool:
        return self._is_team_yellow

    @property
    def game_obj(self) -> Game:
        return self._game_obj

    @property
    def debug(self) -> bool:
        return self._debug
