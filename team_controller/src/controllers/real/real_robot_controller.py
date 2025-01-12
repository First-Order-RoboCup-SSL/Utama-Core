from math import ceil
from serial import Serial
from typing import Union, Optional, Dict, List

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
        self._rbt_cmd_size = self._get_byte_size(
            SERIAL_BIT_SIZES["out"]
        )  # packet size for one robot
        self._quant_dict = (
            self._generate_quant_dict()
        )  # generate values commonly used to quantise output
        self._out_packet = self._empty_command()
        self._in_packet_size = self._get_byte_size(SERIAL_BIT_SIZES["in"])
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
        q_command = self._quantise_command(command)
        command_buffer = self._generate_command_buffer(q_command)
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

    def _generate_quant_dict(self) -> Dict:
        """
        Generates a dictionary of the maximum and minimum quantised values for each key in the SERIAL_BIT_SIZES dictionary.
        """
        quant_dict = {}
        for key, bit_size in SERIAL_BIT_SIZES["out"].items():
            n_bits = bit_size[0]
            signed = True if bit_size[1] == "s" else False
            if signed:
                half_range = 2 ** (n_bits - 1)
                max_quantised = half_range - 1
                min_quantised = -half_range
            else:
                max_quantised = 2**n_bits - 1
                min_quantised = 0
            quant_dict[key] = (max_quantised, min_quantised)
        return quant_dict

    def _quantise_command(self, command: RobotCommand) -> RobotCommand:
        """
        Quantizes the robot command to the appropriate bit size.
        """
        q_angular_vel = self._quantise(
            command.angular_vel, MAX_ANGULAR_VEL, *self._quant_dict["angular_vel"]
        )
        q_local_forward_vel = self._quantise(
            command.local_forward_vel, MAX_VEL, *self._quant_dict["local_forward_vel"]
        )
        q_local_left_vel = self._quantise(
            command.local_left_vel, MAX_VEL, *self._quant_dict["local_left_vel"]
        )
        q_kick = 1 if command.kick > 0 else 0
        q_chip = 1 if command.chip > 0 else 0
        q_dribble = 1 if command.dribble > 0 else 0

        return RobotCommand(
            local_forward_vel=q_local_forward_vel,
            local_left_vel=q_local_left_vel,
            angular_vel=q_angular_vel,
            kick=q_kick,
            chip=q_chip,
            dribble=q_dribble,
        )

    def _quantise(
        self,
        value: float,
        max_actual: float,
        max_quantised: int,
        min_quantised: int,
    ) -> int:
        """
        Normalize a floating-point value and map it to a quantized twos-complement integer range.

        Note the asymmetry in the quantization due to the natural limit of signed values.

        Args:
            value (float): The input value to be quantized.
            max_actual (float): The maximum possible absolute value for normalization.
            max_quantised (int): The upper bound of the quantized range.
            min_quantised (int): The lower bound of the quantized range.

        Returns:
            int: The quantized integer value (two's complement: instead of -4 return its complement).

        Raises:
            AssertionError: If `max_actual` is zero.
        """
        assert max_actual != 0
        assert value <= max_actual and value >= -max_actual

        norm_value = value / max_actual
        if norm_value >= 0:
            quantised = round(norm_value * max_quantised)
        else:
            quantised = -((2 * min_quantised) + round(norm_value * min_quantised))
        return quantised

    def _generate_command_buffer(self, q_command: RobotCommand) -> bytes:
        """
        Generates the command buffer to be sent to the robot.
        """
        out_bit_sizes = SERIAL_BIT_SIZES["out"]
        angular_vel_buffer = (
            f'{q_command.angular_vel:0{out_bit_sizes["angular_vel"][0]}b}'
        )
        local_forward_vel_buffer = (
            f'{q_command.local_forward_vel:0{out_bit_sizes["local_forward_vel"][0]}b}'
        )
        local_left_vel_buffer = (
            f'{q_command.local_left_vel:0{out_bit_sizes["local_left_vel"][0]}b}'
        )
        kick_buffer = f'{q_command.kick:0{out_bit_sizes["kicker_bottom"][0]}b}'
        chip_buffer = f'{q_command.chip:0{out_bit_sizes["kicker_top"][0]}b}'
        dribble_buffer = f'{q_command.dribble:0{out_bit_sizes["dribbler"][0]}b}'

        command_buffer = "".join(
            [
                angular_vel_buffer,
                local_forward_vel_buffer,
                local_left_vel_buffer,
                kick_buffer,
                chip_buffer,
                dribble_buffer,
            ]
        )

        assert len(command_buffer) == self._rbt_cmd_size * 8

        return int(command_buffer, 2).to_bytes(
            self._rbt_cmd_size, byteorder=ENDIAN, signed=False
        )

    def _get_byte_size(self, bit_dict: dict) -> int:
        """
        Returns the byte size of the dictionary. (ceil divide)
        """
        return ceil(sum(value[0] for value in bit_dict.values()) / 8)

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
