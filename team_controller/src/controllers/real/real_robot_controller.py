from math import ceil
from serial import Serial
from typing import Union, Optional, Dict, List
from math import degrees
import warnings
import numpy as np

from entities.data.command import RobotCommand, RobotInfo
from entities.game import Game

from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from team_controller.src.config.settings import (
    MAX_ANGULAR_VEL,
    MAX_VEL,
    BAUD_RATE,
    PORT,
    TIMEOUT,
    ENDIAN,
    SERIAL_BIT_SIZES,
)
import logging

logger = logging.getLogger(__name__)


class RealRobotController(AbstractRobotController):
    """
    Robot Controller for Real Robots.
    """

    def __init__(self, is_team_yellow: bool, game_obj: Game):
        self._is_team_yellow = is_team_yellow
        self._game_obj = game_obj
        self._serial = Serial(port=PORT, baudrate=BAUD_RATE, timeout=TIMEOUT)
        self._rbt_cmd_size = ceil(
            sum(SERIAL_BIT_SIZES["out"].values()) / 8
        )  # packet size for one robot
        self._out_packet = self._empty_command()
        self._in_packet_size = ceil(sum(SERIAL_BIT_SIZES["in"].values()) / 8)
        self._robots_info: List[RobotInfo] = [None] * 6

        logger.debug(
            f"Serial port: {PORT} opened with baudrate: {BAUD_RATE} and timeout {TIMEOUT}"
        )

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).
        """
        self._serial.write(self.out_packet)
        data_in = self._serial.read(self._in_packet_size)
        self._populate_robots_info(data_in)

        self._out_packet = self._empty_command()  # flush the out_packet

    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """
        Adds robot commands to the packet to be sent to the robot.
        """
        super().add_robot_commands(robot_commands, robot_id)

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """
        Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick', 'chip', 'dribble'.
        """
        c_command = self._convert_float_command(command)
        command_buffer = self._generate_command_buffer(c_command)
        start_idx = robot_id * self._rbt_cmd_size
        self._out_packet[start_idx : start_idx + self._rbt_cmd_size] = command_buffer

    def robot_has_ball(self, robot_id):
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
            logger.debug(f"Robot: {robot_id}: HAS the Ball")
            return True
        else:
            return False

    def _populate_robots_info(self, data_in: bytes) -> None:
        """
        Populates the robots_info list with the data received from the robots.

        # TODO It's a bit awkward now because we haven't confirmed the return packet size and there's likely some spare space in the packet
        # assumption now is packet of 1 byte, bits[7] to bit[2] are has_ball boolean in order of id 0 to 5. bit[1] to bit[0] are reserved
        """
        for i in range(6):
            has_ball = False
            if data_in[0] & 0b10000000:
                has_ball = True
            info = RobotInfo(has_ball=has_ball)
            self._robots_info[i] = info
            data_in = data_in << 1  # shift to the next robot's data

    def _generate_command_buffer(self, c_command: RobotCommand) -> bytes:
        """
        Generates the command buffer to be sent to the robot.
        """
        out_bit_sizes = SERIAL_BIT_SIZES["out"]
        local_forward_vel_buffer = (
            f'{c_command.local_forward_vel:0{out_bit_sizes["local_forward_vel"]}b}'
        )
        local_left_vel_buffer = (
            f'{c_command.local_left_vel:0{out_bit_sizes["local_left_vel"]}b}'
        )
        angular_vel_buffer = f'{c_command.angular_vel:0{out_bit_sizes["angular_vel"]}b}'
        kick_buffer = f'{c_command.kick:0{out_bit_sizes["kicker_bottom"]}b}'
        chip_buffer = f'{c_command.chip:0{out_bit_sizes["kicker_top"]}b}'
        dribble_buffer = f'{c_command.dribble:0{out_bit_sizes["dribbler"]}b}'
        spare_buffer = f'{0:0{out_bit_sizes["spare"]}b}'

        command_buffer = "".join(
            [
                local_forward_vel_buffer,
                local_left_vel_buffer,
                angular_vel_buffer,
                kick_buffer,
                chip_buffer,
                dribble_buffer,
                spare_buffer,
            ]
        )

        assert len(command_buffer) == self._rbt_cmd_size * 8

        return int(command_buffer, 2).to_bytes(
            self._rbt_cmd_size, byteorder=ENDIAN, signed=False
        )

    def _convert_float_command(self, command: RobotCommand) -> RobotCommand:
        """
        Prepares the float values in the command to be formatted to binary in the buffer.

        Also converts angular velocity to degrees per second.
        """

        angular_vel = command.angular_vel
        local_forward_vel = command.local_forward_vel
        local_left_vel = command.local_left_vel

        if abs(command.angular_vel) > MAX_ANGULAR_VEL:
            warnings.warn(
                f"Angular velocity for robot {command.robot_id} is greater than the maximum angular velocity. Clipping to {MAX_ANGULAR_VEL}."
            )
            angular_vel = (
                MAX_ANGULAR_VEL if command.angular_vel > 0 else -MAX_ANGULAR_VEL
            )

        if abs(command.local_forward_vel) > MAX_VEL:
            warnings.warn(
                f"Local forward velocity for robot {command.robot_id} is greater than the maximum velocity. Clipping to {MAX_VEL}."
            )
            local_forward_vel = MAX_VEL if command.local_forward_vel > 0 else -MAX_VEL

        if abs(command.local_left_vel) > MAX_VEL:
            warnings.warn(
                f"Local left velocity for robot {command.robot_id} is greater than the maximum velocity. Clipping to {MAX_VEL}."
            )
            local_left_vel = MAX_VEL if command.local_left_vel > 0 else -MAX_VEL

        out_bit_sizes = SERIAL_BIT_SIZES["out"]
        command = RobotCommand(
            local_forward_vel=self._convert_float(
                local_forward_vel, out_bit_sizes["local_forward_vel"]
            ),
            local_left_vel=self._convert_float(
                local_left_vel, out_bit_sizes["local_left_vel"]
            ),
            angular_vel=self._convert_float(
                degrees(angular_vel), out_bit_sizes["angular_vel"]
            ),
            kick=command.kick,
            chip=command.chip,
            dribble=command.dribble,
        )
        return command

    def _convert_float(self, val: float, float_size: int) -> int:
        """
        Converts a float to an unsigned integer using the specified float size.
        This allows us to format it to binary and send it to the robot.
        """
        if float_size == 16:
            float_val = np.float16(val)
        elif float_size == 32:
            float_val = np.float32(val)
        else:
            raise ValueError(f"Invalid float size: {float_size}")

        return float_val.view(np.uint16)

    def _empty_command(self) -> bytearray:
        return bytearray(self._rbt_cmd_size * 6)

    @property
    def is_team_yellow(self) -> bool:
        return self._is_team_yellow

    @property
    def game_obj(self) -> Game:
        return self._game_obj

    @property
    def serial(self) -> Serial:
        return self._serial

    @property
    def rbt_cmd_size(self) -> int:
        return self._rbt_cmd_size

    @property
    def out_packet(self) -> bytearray:
        return self._out_packet

    @property
    def in_packet_size(self) -> int:
        return self._in_packet_size
