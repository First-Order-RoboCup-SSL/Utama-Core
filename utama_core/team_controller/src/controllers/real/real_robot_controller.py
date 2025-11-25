import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
from serial import EIGHTBITS, PARITY_EVEN, STOPBITS_TWO, Serial

from utama_core.config.robot_params import REAL_PARAMS
from utama_core.config.settings import BAUD_RATE, PORT, TIMEOUT
from utama_core.entities.data.command import RobotCommand, RobotResponse
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

logger = logging.getLogger(__name__)

# NB: A major assumption is that the robot IDs are 0-5 for the friendly team.
MAX_VEL = REAL_PARAMS.MAX_VEL
MAX_ANGULAR_VEL = REAL_PARAMS.MAX_ANGULAR_VEL


class RealRobotController(AbstractRobotController):
    """Robot Controller for Real Robots.

    Args:
        is_team_yellow (bool): True if the team is yellow, False if the team is blue.
        n_robots (int): The number of robots in the team. Directly affects output buffer size. Default is 6.
    """

    def __init__(self, is_team_yellow: bool, n_friendly: int):
        super().__init__(is_team_yellow, n_friendly)
        self._serial_port = self._init_serial()
        self._rbt_cmd_size = 10  # packet size for one robot
        self._out_packet = self._empty_command()
        self._in_packet_size = 1  # size of the feedback packet received from the robots
        self._robots_info: List[RobotResponse] = [None] * self._n_friendly

        logger.debug(f"Serial port: {PORT} opened with baudrate: {BAUD_RATE} and timeout {TIMEOUT}")

    def send_robot_commands(self) -> None:
        """Sends the robot commands to the appropriate team (yellow or blue)."""
        # print(list(self.out_packet))
        # binary_representation = [f"{byte:08b}" for byte in self.out_packet]
        # print(binary_representation)
        self._serial_port.write(self.out_packet)
        self._serial_port.read_all()
        # data_in = self._serial.read_all()
        # print(data_in)

        # TODO: add receiving feedback from the robots

        self._out_packet = self._empty_command()  # flush the out_packet

    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        """Adds robot commands to the packet to be sent to the robot."""
        super().add_robot_commands(robot_commands, robot_id)

    def _add_robot_command(self, command: RobotCommand, robot_id: int) -> None:
        """Adds a robot command to the out_packet.

        Args:
            robot_id (int): The ID of the robot.
            command (RobotCommand): A named tuple containing the robot command with keys: 'local_forward_vel', 'local_left_vel', 'angular_vel', 'kick', 'chip', 'dribble'.
        """
        c_command = self._convert_float16_command(robot_id, command)
        command_buffer = self._generate_command_buffer(robot_id, c_command)
        print(command_buffer)
        start_idx = robot_id * self._rbt_cmd_size + 1  # account for the start frame byte
        self._out_packet[start_idx : start_idx + self._rbt_cmd_size] = (
            command_buffer  # +1 to account for start frame byte
        )

    # def _populate_robots_info(self, data_in: bytes) -> None:
    #     """
    #     Populates the robots_info list with the data received from the robots.
    #     """
    #     for i in range(self._n_friendly):
    #         has_ball = False
    #         if data_in[0] & 0b10000000:
    #             has_ball = True
    #         info = RobotInfo(has_ball=has_ball)
    #         self._robots_info[i] = info
    #         data_in = data_in << 1  # shift to the next robot's data

    def _generate_command_buffer(self, robot_id: int, c_command: RobotCommand) -> bytes:
        """Generates the command buffer to be sent to the robot."""
        assert robot_id < 6, "Invalid robot_id. Must be between 0 and 5."

        # Combine first 6 bytes of velocities
        # packet = bytearray(
        #     [
        #         robot_id & 0xFF,  # Robot ID
        #         (c_command.local_forward_vel >> 8) & 0xFF,  # Forward velocity high byte
        #         c_command.local_forward_vel & 0xFF,  # Forward velocity low byte
        #         (c_command.local_left_vel >> 8) & 0xFF,  # Left velocity high byte
        #         c_command.local_left_vel & 0xFF,  # Left velocity low byte
        #         (c_command.angular_vel >> 8) & 0xFF,  # Angular velocity high byte
        #         c_command.angular_vel & 0xFF,  # Angular velocity low byte
        #     ]
        # )
        packet = bytearray(
            [
                robot_id & 0xFF,  # Robot ID
                c_command.local_forward_vel & 0xFF,  # Forward velocity low byte
                (c_command.local_forward_vel >> 8) & 0xFF,  # Forward velocity high byte
                c_command.local_left_vel & 0xFF,  # Left velocity low byte
                (c_command.local_left_vel >> 8) & 0xFF,  # Left velocity high byte
                c_command.angular_vel & 0xFF,  # Angular velocity low byte
                (c_command.angular_vel >> 8) & 0xFF,  # Angular velocity high byte
            ]
        )

        dribbler_speed = 0
        if c_command.dribble:
            dribbler_speed = 0xC000  # set bits 15:14 to 11
            dribbler_speed |= 4095 & 0x3FFF  # set bits 13:0 to 4095

        packet.extend(
            [
                (dribbler_speed >> 8) & 0xFF,  # Dribbler high byte
                dribbler_speed & 0xFF,  # Dribbler low byte
            ]
        )

        kicker_byte = 0
        if c_command.kick:
            kicker_byte |= 0xF0  # upper kicker full power
        if c_command.chip:
            kicker_byte |= 0x0F
        packet.append(kicker_byte)  # Kicker controls  # Frame end

        # packet_str = " ".join(f"{byte:08b}" for byte in packet)

        return packet

    def _convert_float16_command(self, robot_id, command: RobotCommand) -> RobotCommand:
        """Prepares the float values in the command to be formatted to binary in the buffer.

        Also converts angular velocity to degrees per second.
        """

        angular_vel = command.angular_vel
        local_forward_vel = command.local_forward_vel
        local_left_vel = command.local_left_vel

        if abs(command.angular_vel) > MAX_ANGULAR_VEL:
            warnings.warn(
                f"Angular velocity for robot {robot_id} is greater than the maximum angular velocity. Clipping to {MAX_ANGULAR_VEL}."
            )
            angular_vel = MAX_ANGULAR_VEL if command.angular_vel > 0 else -MAX_ANGULAR_VEL
        # TODO put back to max_vel
        if abs(command.local_forward_vel) > 0.8:
            warnings.warn(
                f"Local forward velocity for robot {robot_id} is greater than the maximum velocity. Clipping to {MAX_VEL}."
            )
            local_forward_vel = MAX_VEL if command.local_forward_vel > 0 else -MAX_VEL

        if abs(command.local_left_vel) > MAX_VEL:
            warnings.warn(
                f"Local left velocity for robot {robot_id} is greater than the maximum velocity. Clipping to {MAX_VEL}."
            )
            local_left_vel = MAX_VEL if command.local_left_vel > 0 else -MAX_VEL

        command = RobotCommand(
            local_forward_vel=self._float16_rep(local_forward_vel),
            local_left_vel=self._float16_rep(local_left_vel),
            angular_vel=self._float16_rep(angular_vel),
            kick=command.kick,
            chip=command.chip,
            dribble=command.dribble,
        )
        return command

    def _float16_rep(self, value: float) -> np.uint16:
        """Converts a float, flattens it to float 16 and represented as uint16 value for transmission."""
        return np.float16(value).view(np.uint16)

    def _sanitise_float(self, val: float) -> float:
        """Map NaN/±inf to 0.0 to avoid propagating bad values."""
        if not np.isfinite(val):
            return 0.0
        return val

    def _empty_command(self) -> bytearray:
        if not hasattr(self, "_cached_empty_command"):
            commands = bytearray()
            for robot_id in range(self._n_friendly):
                cmd = bytearray([robot_id] + [0] * (self._rbt_cmd_size - 1))  # empty command for each robot
                commands += cmd
            self._cached_empty_command = bytearray([0xAA]) + commands + bytearray([0x55])
        return self._cached_empty_command

    def _init_serial(self) -> Serial:
        """Establish serial connection."""
        try:
            # Open new connection
            serial_port = Serial(
                port=PORT,
                baudrate=BAUD_RATE,
                bytesize=EIGHTBITS,  # 8 data bits
                parity=PARITY_EVEN,  # Even parity (makes it 9 bits total)
                stopbits=STOPBITS_TWO,  # 2 stop bits
                timeout=0.1,
            )
            return serial_port
        except Exception as e:
            raise ConnectionError(f"Could not connect to serial port {PORT}: {e}") from e

    @property
    def is_team_yellow(self) -> bool:
        return self._is_team_yellow

    @property
    def serial_port(self) -> Serial:
        return self._serial_port

    @property
    def rbt_cmd_size(self) -> int:
        return self._rbt_cmd_size

    @property
    def out_packet(self) -> bytearray:
        return self._out_packet

    @property
    def n_friendly(self) -> int:
        return self._n_friendly

    @property
    def in_packet_size(self) -> int:
        return self._in_packet_size


if __name__ == "__main__":
    robot_controller = RealRobotController(is_team_yellow=True, n_friendly=1)
    cmd = RobotCommand(
        local_forward_vel=0.2,
        local_left_vel=0,
        angular_vel=0,
        kick=0,
        chip=0,
        dribble=False,
    )
    for _ in range(15):
        robot_controller.add_robot_commands(cmd, 0)
        # robot_controller.send_robot_commands()
    # for _ in range(10):
    #     robot_controller.add_robot_commands(empty_command(), 0)
    #     robot_controller.send_robot_commands()

    # print(list(robot_controller.out_packet))
    # binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
    # print(binary_representation)
