from serial import Serial
from typing import Union, Optional, Dict, List
import warnings
import numpy as np
import time

from entities.data.command import RobotCommand, RobotResponse
from entities.game import Game

from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from config.settings import (
    MAX_ANGULAR_VEL,
    MAX_VEL,
    BAUD_RATE,
    PORT,
    TIMEOUT,
    AUTH_STR,
    MAX_INITIALIZATION_TIME,
)
import logging

logger = logging.getLogger(__name__)


class RealRobotController(AbstractRobotController):
    """
    Robot Controller for Real Robots.

    Args:
        is_team_yellow (bool): True if the team is yellow, False if the team is blue.
        game_obj (Game): The game object storing all game state data.
        n_robots (int): The number of robots in the team. Directly affects output buffer size. Default is 6.
    """

    def __init__(
        self,
        is_team_yellow: bool,
        game_obj: Game = None,
        n_robots: int = 6,
    ):
        self._is_team_yellow = is_team_yellow
        self._game_obj = game_obj
        self._n_robots = n_robots  # determines buffer size
        self._serial = self._init_serial()

        self._EMPTY_ID = 30  # id to indicate empty buffer: 1110 in control byte
        self._rbt_cmd_size = 8  # packet size for one robot
        self._out_packet = self._empty_command()
        self._in_packet_size = 1  # size of the packet received from the robots
        self._robots_info: List[RobotResponse] = [None] * self._n_robots

        logger.debug(
            f"Serial port: {PORT} opened with baudrate: {BAUD_RATE} and timeout {TIMEOUT}"
        )

    def send_robot_commands(self) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).
        """
        # print(list(self.out_packet))
        # binary_representation = [f"{byte:08b}" for byte in self.out_packet]
        # print(binary_representation)
        self._serial.write(self.out_packet)
        data_in = self._serial.read_all()
        # print(data_in)

        # TODO: this is only for quali: fix this after quali
        if len(data_in) == 1:
            if data_in[0] & 0b01000000:
                self._robots_info[1] = RobotResponse(has_ball=True)
            else:
                self._robots_info[1] = RobotResponse(has_ball=False)
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
        c_command = self._convert_float_command(robot_id, command)
        command_buffer = self._generate_command_buffer(robot_id, c_command)
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

    # def _populate_robots_info(self, data_in: bytes) -> None:
    #     """
    #     Populates the robots_info list with the data received from the robots.
    #     """
    #     for i in range(self._n_robots):
    #         has_ball = False
    #         if data_in[0] & 0b10000000:
    #             has_ball = True
    #         info = RobotInfo(has_ball=has_ball)
    #         self._robots_info[i] = info
    #         data_in = data_in << 1  # shift to the next robot's data

    def compute_crc(self, data: bytearray) -> int:
        """
        Calculate CRC-8, use 0x07 polynomial.
        这里的计算对 data 中的每个字节进行处理。
        """
        poly = 0x07
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ poly) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    def _generate_command_buffer(self, robot_id: int, c_command: RobotCommand) -> bytes:
        """
        Generates the command buffer to be sent to the robot.
        """
        assert robot_id < 6, "Invalid robot_id. Must be between 0 and 5."

        # Combine first 6 bytes of velocities
        packet = bytearray(
            [
                (c_command.local_forward_vel >> 8) & 0xFF,
                c_command.local_forward_vel & 0xFF,
                (c_command.local_left_vel >> 8) & 0xFF,
                c_command.local_left_vel & 0xFF,
                (c_command.angular_vel >> 8) & 0xFF,
                c_command.angular_vel & 0xFF,
            ]
        )

        # Create control byte
        control_byte = 0
        if c_command.dribble:
            control_byte |= 0x20  # Bit 5
        if c_command.chip:
            control_byte |= 0x40  # Bit 6
        if c_command.kick:
            control_byte |= 0x80  # Bit 7
        robot_id = robot_id & 0x0F  # 5 bits only
        control_byte |= robot_id << 1
        # set last bit as 1 if its the last command
        # TODO: this fails on cases wher we are only communicating with one robot and their id is not 0
        if robot_id == self._n_robots - 1:
            control_byte |= 0x01
        packet.append(control_byte)
        crc = self.compute_crc(packet)
        packet.append(crc)
        
        packet_str = " ".join(f"{byte:08b}" for byte in packet)

        return packet

    def _convert_float_command(self, robot_id, command: RobotCommand) -> RobotCommand:
        """
        Prepares the float values in the command to be formatted to binary in the buffer.

        Also converts angular velocity to degrees per second.
        """

        angular_vel = command.angular_vel
        local_forward_vel = command.local_forward_vel
        local_left_vel = command.local_left_vel

        if abs(command.angular_vel) > MAX_ANGULAR_VEL:
            warnings.warn(
                f"Angular velocity for robot {robot_id} is greater than the maximum angular velocity. Clipping to {MAX_ANGULAR_VEL}."
            )
            angular_vel = (
                MAX_ANGULAR_VEL if command.angular_vel > 0 else -MAX_ANGULAR_VEL
            )
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
        """
        Converts a float, flattens it to float 16 and represented as uint16 value for transmission.
        """
        return np.float16(value).view(np.uint16)

    def _empty_command(self) -> bytearray:
        empty_buffer = bytearray([0] * 6 + [self._EMPTY_ID] + [0])
        empty_last_buffer = bytearray([0] * 6 + [self._EMPTY_ID + 1] + [0])
        return empty_buffer * (self._n_robots - 1) + empty_last_buffer

    def _init_serial(self) -> Serial:
        serial = Serial(port=PORT, baudrate=BAUD_RATE, timeout=TIMEOUT)
        start_t = time.time()
        is_ready = False
        while time.time() - start_t < MAX_INITIALIZATION_TIME:
            if serial.in_waiting > 0:
                line = serial.readline().decode("utf-8").rstrip()
                if line == AUTH_STR:
                    is_ready = True
                    break
                else:
                    print(line)

        if is_ready:
            print("Serial port opened!")
            serial.reset_input_buffer()  # temporary implementation to clear debugging info in input
        else:
            raise ConnectionError("Could not connect: Invalid authentication string!")
        return serial

    @property
    def is_team_yellow(self) -> bool:
        return self._is_team_yellow

    @property
    def game_obj(self) -> Game:
        return self._game_obj

    @property
    def n_robots(self) -> int:
        return self._n_robots

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
    def n_robot(self) -> int:
        return self._n_robots

    @property
    def in_packet_size(self) -> int:
        return self._in_packet_size


if __name__ == "__main__":
    robot_controller = RealRobotController(
        is_team_yellow=True, game_obj=Game(), n_robots=2
    )
    cmd = RobotCommand(
        local_forward_vel=0.2,
        local_left_vel=0,
        angular_vel=0,
        kick=0,
        chip=0,
        dribble=False,
    )
    robot_controller.add_robot_commands(cmd, 0)
    print(list(robot_controller.out_packet))
    binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
    print(binary_representation)
