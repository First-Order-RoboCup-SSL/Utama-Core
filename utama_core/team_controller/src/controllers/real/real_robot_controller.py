import dataclasses
import logging
import time
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
from serial import EIGHTBITS, PARITY_EVEN, STOPBITS_TWO, Serial

from utama_core.config.robot_params import REAL_PARAMS
from utama_core.config.settings import (
    BAUD_RATE,
    KICKER_COOLDOWN_TIMESTEPS,
    KICKER_PERSIST_TIMESTEPS,
    PORT,
    TIMESTEP,
)
from utama_core.entities.data.command import RobotCommand, RobotResponse
from utama_core.skills.src.utils.move_utils import empty_command
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

logger = logging.getLogger(__name__)

MAX_VEL = REAL_PARAMS.MAX_VEL
MAX_ANGULAR_VEL = REAL_PARAMS.MAX_ANGULAR_VEL


@dataclasses.dataclass
class KickTrackerEntry:
    remaining_persist: int
    remaining_cooldown: int
    is_kick: bool  # True for kick, False for chip


class RealRobotController(AbstractRobotController):
    """Robot Controller for Real Robots.

    Args:
        is_team_yellow (bool): True if the team is yellow, False if the team is blue.
        n_robots (int): The number of robots in the team. Directly affects output buffer size. Default is 6.
    """

    HEADER = 0xAA
    TAIL = 0x55

    def __init__(self, is_team_yellow: bool, n_friendly: int):
        super().__init__(is_team_yellow, n_friendly)
        self._serial_port = self._init_serial()
        self._rx_buffer = bytearray()
        self._rbt_cmd_size = 10  # packet size (bytes) for one robot
        self._robot_info_size = 2  # number of bytes in feedback packet for all robots
        self._out_packet = self._empty_command()
        self._in_packet_size = 1  # size of the feedback packet received from the robots
        logger.debug(f"Serial port: {PORT} opened with baudrate: {BAUD_RATE}.")
        self._assigned_mapping = {}  # mapping of robot_id to index in the out_packet

        # track last kick time for each robot to transmit kick as HIGH for n timesteps after command
        self._kicker_tracker: Dict[int, KickTrackerEntry] = {}

    def get_robots_responses(self) -> Optional[Dict[int, RobotResponse]]:
        """Reads the robot responses from the serial port and returns a dictionary mapping robot IDs to their responses."""
        # For now, we are not processing feedback from the robots, but this method can be implemented similarly to the GRSimRobotController if needed in the future.
        self._serial_port.read_all()  # Clear the input buffer to avoid overflow
        return None

    # TODO: implement when feedback is actually working
    def get_robots_responses_real(self) -> Optional[Dict[int, RobotResponse]]:
        FRAME_SIZE = 1 + self._robot_info_size + 1

        # --- Step 1: Non-blocking read ---
        waiting = self._serial_port.in_waiting
        if waiting:
            self._rx_buffer.extend(self._serial_port.read(waiting))

        latest_payload = None

        # --- Step 2: Forward scan to find the newest COMPLETE frame ---
        while len(self._rx_buffer) >= FRAME_SIZE:
            if self._rx_buffer[0] != self.HEADER:
                # Fast-forward to the next possible header to save CPU cycles
                next_header = self._rx_buffer.find(bytes([self.HEADER]))
                if next_header == -1:
                    self._rx_buffer.clear()
                    break
                self._rx_buffer = self._rx_buffer[next_header:]
                continue

            # We have a header and enough bytes. Validate tail.
            if self._rx_buffer[FRAME_SIZE - 1] == self.TAIL:
                # Valid frame found! Save it, but keep looping to see if there's a newer one.
                latest_payload = self._rx_buffer[1 : FRAME_SIZE - 1]
                self._rx_buffer = self._rx_buffer[FRAME_SIZE:]
            else:
                # Fake header. Drop it so we don't get stuck, and resync.
                self._rx_buffer.pop(0)

        # --- Step 3: If no complete frames were found, check for a partial one to block for ---
        if not latest_payload and len(self._rx_buffer) > 0 and self._rx_buffer[0] == self.HEADER:
            remaining = FRAME_SIZE - len(self._rx_buffer)
            chunk = self._serial_port.read(remaining)  # Blocks
            if chunk:
                self._rx_buffer.extend(chunk)

            # Validate the newly completed frame
            if len(self._rx_buffer) == FRAME_SIZE and self._rx_buffer[FRAME_SIZE - 1] == self.TAIL:
                latest_payload = self._rx_buffer[1 : FRAME_SIZE - 1]
                self._rx_buffer.clear()
            else:
                # Fake header or timeout. Drop the bad byte.
                self._rx_buffer.pop(0)

        return None  # Placeholder until we implement actual parsing of latest_payload

    def send_robot_commands(self) -> None:
        """Sends the robot commands to the appropriate team (yellow or blue)."""
        # print(list(self.out_packet))
        # binary_representation = [f"{byte:02x}" for byte in self.out_packet]
        # print(binary_representation)
        if len(self._assigned_mapping) != self._n_friendly:
            warnings.warn(
                f"Only {len(self._assigned_mapping)} out of {self._n_friendly} robots have been assigned commands. Sending incomplete command packet."
            )
        self._serial_port.write(self._out_packet)

        ### update kick and chip trackers. We persist the kick/chip command for KICKER_PERSIST_TIMESTEPS
        ### this feature is to combat packet loss and to ensure the robot does not kick within its cooldown period
        ### Embedded only registers rising edge of kick/chip command
        # TODO: this logic has to be reconsidered. Solution for qualification now.

        for robot_id in list(self._kicker_tracker.keys()):
            if self._kicker_tracker[robot_id].remaining_cooldown > 1:
                self._kicker_tracker[robot_id].remaining_cooldown -= 1

                if self._kicker_tracker[robot_id].remaining_persist > 0:
                    self._kicker_tracker[robot_id].remaining_persist -= 1
            else:
                # remove kicker tracker entry once cooldown is over
                del self._kicker_tracker[robot_id]

        self._out_packet = self._empty_command()  # flush the out_packet
        self._assigned_mapping = {}  # reset assigned mapping

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
        if robot_id in self._assigned_mapping:
            warnings.warn(
                f"Robot ID {robot_id} has already been assigned a command in this cycle. Overwriting previous command."
            )
            start_idx = self._assigned_mapping[robot_id] * (
                self._rbt_cmd_size + 2
            )  # account for start and end bytes per robot

        elif len(self._assigned_mapping) >= self._n_friendly:
            warnings.warn(
                f"All {self._n_friendly} robots have already been assigned commands in this cycle. Ignoring command for robot ID {robot_id}."
            )
            return
        else:
            assigned_idx = len(self._assigned_mapping)
            start_idx = assigned_idx * (self._rbt_cmd_size + 2)  # account for start and end bytes per robot
            self._assigned_mapping[robot_id] = assigned_idx

        c_command = self._convert_float16_command(robot_id, command)
        command_buffer = self._generate_command_buffer(robot_id, c_command)
        self._out_packet[start_idx + 1 : start_idx + self._rbt_cmd_size + 1] = command_buffer

        if command.kick and robot_id not in self._kicker_tracker:
            self._kicker_tracker[robot_id] = KickTrackerEntry(
                remaining_persist=KICKER_PERSIST_TIMESTEPS,
                remaining_cooldown=KICKER_COOLDOWN_TIMESTEPS,
                is_kick=True,
            )
        # if for some reason we are kicking and chipping at the same time, we prioritise kick
        if command.chip and robot_id not in self._kicker_tracker:
            self._kicker_tracker[robot_id] = KickTrackerEntry(
                remaining_persist=KICKER_PERSIST_TIMESTEPS,
                remaining_cooldown=KICKER_COOLDOWN_TIMESTEPS,
                is_kick=False,
            )

    def _generate_command_buffer(self, robot_id: int, c_command: RobotCommand) -> bytes:
        """Generates the command buffer to be sent to the robot."""
        assert robot_id < 6, "Invalid robot_id. Must be between 0 and 5."

        # endianness: little endian
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

        #### KICKER LOGIC ####
        # cooldown ensures that we do not resend kick/chip command within cooldown period
        # persist ensures that we resend kick/chip command for n timesteps after initial command
        # embedded detects rising edge of kick/chip command only
        ######################

        kicker_byte = 0
        tracker = self._kicker_tracker.get(robot_id)

        # If tracker_entry exists but persistence expired → send empty kicker byte and return
        if tracker and tracker.remaining_persist <= 0:
            packet.append(kicker_byte)
            return packet

        # Decide whether we're kicking or chipping
        kick_active = c_command.kick or (tracker and tracker.remaining_persist > 0 and tracker.is_kick)

        chip_active = c_command.chip or (tracker and tracker.remaining_persist > 0 and not tracker.is_kick)

        if kick_active:
            kicker_byte |= 0xF0  # upper kicker full power
        elif chip_active:
            kicker_byte |= 0x0F

        packet.append(kicker_byte)

        # packet_str = " ".join(f"{byte:08b}" for byte in packet)

        return packet

    def _convert_float16_command(self, robot_id, command: RobotCommand) -> RobotCommand:
        """Prepares the float values in the command to be formatted to binary in the buffer.

        Also converts angular velocity to degrees per second.
        """
        angular_vel = self._sanitise_float(command.angular_vel)
        local_forward_vel = self._sanitise_float(command.local_forward_vel)
        local_left_vel = self._sanitise_float(command.local_left_vel)

        if abs(angular_vel) > MAX_ANGULAR_VEL:
            warnings.warn(
                f"Angular velocity for robot {robot_id} is greater than the maximum angular velocity. Clipping to {MAX_ANGULAR_VEL}."
            )
            angular_vel = MAX_ANGULAR_VEL if angular_vel > 0 else -MAX_ANGULAR_VEL
        if abs(local_forward_vel) > MAX_VEL:
            warnings.warn(
                f"Local forward velocity for robot {robot_id} is greater than the maximum velocity. Clipping to {MAX_VEL}."
            )
            local_forward_vel = MAX_VEL if local_forward_vel > 0 else -MAX_VEL
        if abs(local_left_vel) > MAX_VEL:
            warnings.warn(
                f"Local left velocity for robot {robot_id} is greater than the maximum velocity. Clipping to {MAX_VEL}."
            )
            local_left_vel = MAX_VEL if local_left_vel > 0 else -MAX_VEL

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
            INVALID_RBT_ID = 0xFF
            commands = bytearray()
            for _ in range(self._n_friendly):
                # Empty command for each robot:
                #  - 0xAA: start byte
                #  - command buffer of length self._rbt_cmd_size, where:
                #      * first byte is robot ID (here INVALID_RBT_ID)
                #      * remaining (self._rbt_cmd_size - 1) bytes are zeros
                #  - 0x55: end byte
                cmd = bytearray([self.HEADER] + [INVALID_RBT_ID] + [0] * (self._rbt_cmd_size - 1) + [self.TAIL])
                commands.extend(cmd)
            self._cached_empty_command = commands
        return self._cached_empty_command.copy()

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
                timeout=0,
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
    for _ in range(100):
        robot_controller.add_robot_commands(cmd, 0)
        robot_controller.send_robot_commands()
        time.sleep(TIMESTEP)
    for _ in range(10):
        robot_controller.add_robot_commands(empty_command(), 0)
        robot_controller.send_robot_commands()
        time.sleep(TIMESTEP)

    # print(list(robot_controller.out_packet))
    # binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
    # print(binary_representation)
