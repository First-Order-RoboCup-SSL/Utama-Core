import threading
import time
import queue
from typing import Tuple, Optional, List

from entities.referee.referee_command import RefereeCommand
from entities.referee.stage import Stage
from entities.game.team_info import TeamInfo
from entities.data.referee import RefereeData
from team_controller.src.utils import network_manager
from config.settings import MULTICAST_GROUP_REFEREE, REFEREE_PORT

from team_controller.src.generated_code.ssl_gc_referee_message_pb2 import Referee
import logging

logger = logging.getLogger(__name__)


class RefereeMessageReceiver:
    """
    A class responsible for receiving and managing referee messages in a multi-robot game environment.
    The class interfaces with a network manager to receive packets, which contain game state information,
    and updates the internal data structures accordingly.

    Args:
        ip (str): The IP address for receiving multicast referee data. Defaults to MULTICAST_GROUP_REFEREE.
        port (int): The port for receiving referee data. Defaults to REFEREE_PORT.
    """

    def __init__(
        self,
        message_queue: queue.SimpleQueue,
        ip=MULTICAST_GROUP_REFEREE,
        port=REFEREE_PORT,
        debug=False,
    ):
        super().__init__(message_queue)

        self.net = network_manager.NetworkManager(address=(ip, port), bind_socket=True)
        self.prev_command_counter = -1
        self.command_history = []
        self.referee = Referee()  # Initialize a single Referee object
        self.old_serialized_data = None
        self.time_received = None
        self.lock = threading.Lock()
        self.update_event = threading.Event()
        self.message_queue = message_queue

        # Initialize state variables
        self.stage = None
        self.command = None
        self.time_sent = None
        self.stage_time_left = None
        self.command_counter = None
        self.command_timestamp = None
        self.yellow_info = TeamInfo("yellow")
        self.blue_info = TeamInfo("blue")

    def string_from_stage(self, stage: int) -> str:
        """
        Converts a stage enum value to a string.

        Args:
            stage (int): The stage enum value.

        Returns:
            str: The string representation of the stage.
        """
        stage_map = {
            Referee.NORMAL_FIRST_HALF_PRE: "Normal First Half Prep",
            Referee.NORMAL_FIRST_HALF: "Normal First Half",
            Referee.NORMAL_HALF_TIME: "Normal Half Time",
            Referee.NORMAL_SECOND_HALF_PRE: "Normal Second Half Prep",
            Referee.NORMAL_SECOND_HALF: "Normal Second Half",
            Referee.EXTRA_TIME_BREAK: "Extra Time Break",
            Referee.EXTRA_FIRST_HALF_PRE: "Extra First Half Prep",
            Referee.EXTRA_FIRST_HALF: "Extra First Half",
            Referee.EXTRA_HALF_TIME: "Extra Half Time",
            Referee.EXTRA_SECOND_HALF_PRE: "Extra Second Half Prep",
            Referee.EXTRA_SECOND_HALF: "Extra Second Half",
            Referee.PENALTY_SHOOTOUT_BREAK: "Penalty Shootout Break",
            Referee.PENALTY_SHOOTOUT: "Penalty Shootout",
            Referee.POST_GAME: "Post Game",
        }
        return stage_map.get(stage, "")

    def string_from_command(self, command: int) -> str:
        """
        Converts a command enum value to a string.

        Args:
            command (int): The command enum value.

        Returns:
            str: The string representation of the command.
        """
        command_map = {
            Referee.HALT: "Halt",
            Referee.STOP: "Stop",
            Referee.NORMAL_START: "Normal Start",
            Referee.FORCE_START: "Force Start",
            Referee.PREPARE_KICKOFF_YELLOW: "Yellow Kickoff Prep",
            Referee.PREPARE_KICKOFF_BLUE: "Blue Kickoff Prep",
            Referee.PREPARE_PENALTY_YELLOW: "Yellow Penalty Prep",
            Referee.PREPARE_PENALTY_BLUE: "Blue Penalty Prep",
            Referee.DIRECT_FREE_YELLOW: "Direct Yellow Free Kick",
            Referee.DIRECT_FREE_BLUE: "Direct Blue Free Kick",
            Referee.INDIRECT_FREE_YELLOW: "Indirect Yellow Free Kick",
            Referee.INDIRECT_FREE_BLUE: "Indirect Blue Free Kick",
            Referee.TIMEOUT_YELLOW: "Timeout Yellow",
            Referee.TIMEOUT_BLUE: "Timeout Blue",
            Referee.GOAL_YELLOW: "Goal Yellow",
            Referee.GOAL_BLUE: "Goal Blue",
            Referee.BALL_PLACEMENT_YELLOW: "Ball Placement Yellow",
            Referee.BALL_PLACEMENT_BLUE: "Ball Placement Blue",
        }
        return command_map.get(command, "")

    def _serialize_relevant_fields(self, data: bytes) -> bytes:
        """
        Serialize relevant fields of the referee message, excluding `packet_timestamp` and `stage_time_left`.

        Args:
            data (bytes): The raw data received from the network.

        Returns:
            bytes: The serialized data with excluded fields set to default values.
        """
        # Create a shallow copy of the message
        message_copy = Referee()
        message_copy.ParseFromString(data)

        # Exclude `packet_timestamp` and `stage_time_left` by setting them to a default value
        message_copy.packet_timestamp = 0
        message_copy.stage_time_left = 0

        # Serialize the message to a byte string
        return message_copy.SerializeToString()

    def _update_data(self, referee_packet: Referee) -> None:
        """
        Update the internal data structures with the new referee packet.

        Args:
            referee_packet (Referee): The referee packet containing game state information.
        """
        self.referee = referee_packet
        self.update_event.set()  # Signal that an update has occurred.

        # Update state variables
        self.stage = Stage.from_id(referee_packet.stage)
        self.command = RefereeCommand.from_id(referee_packet.command)
        self.time_sent = (
            referee_packet.packet_timestamp / 1e6
        )  # Convert microseconds to seconds
        self.stage_time_left = (
            referee_packet.stage_time_left / 1e3
        )  # Convert milliseconds to seconds
        self.command_counter = referee_packet.command_counter
        self.command_timestamp = (
            referee_packet.command_timestamp / 1e6
        )  # Convert microseconds to seconds
        self.yellow_info.parse_referee_packet(referee_packet.yellow)
        self.blue_info.parse_referee_packet(referee_packet.blue)

        # Construct the designated position tuple if available
        designated_position = None
        if referee_packet.HasField("designated_position"):
            designated_position = (
                referee_packet.designated_position.x,
                referee_packet.designated_position.y,
            )

        # Construct the RefereeData instance
        referee_data = RefereeData(
            source_identifier=referee_packet.source_identifier,
            time_sent=self.time_sent,
            time_received=self.time_received,
            referee_command=self.command,
            referee_command_timestamp=self.command_timestamp,
            stage=self.stage,
            stage_time_left=self.stage_time_left,
            blue_team=self.blue_info,
            yellow_team=self.yellow_info,
            designated_position=designated_position,
            blue_team_on_positive_half=referee_packet.blue_team_on_positive_half,
            next_command=(
                RefereeCommand.from_id(referee_packet.next_command)
                if referee_packet.HasField("next_command")
                else None
            ),
            current_action_time_remaining=(
                referee_packet.current_action_time_remaining
                if referee_packet.HasField("current_action_time_remaining")
                else None
            ),
        )

        self._message_queue.put_nowait((MessageType.REF, referee_data))

    def check_new_command(self) -> bool:
        """
        Check if a new command has been received and update the command history.

        Returns:
            bool: True if a new command has been received, False otherwise.
        """
        history_length = 5

        if data:
            self.referee.ParseFromString(data)  # Reuse the same object
            if self.referee.command_counter != self.prev_command_counter:
                self.prev_command_counter = self.referee.command_counter
                self.command_history.append(self.referee.command)
                if len(self.command_history) > history_length:
                    self.command_history.pop(0)  # Maintain a fixed-length history
                return True
        return False

    def get_latest_message(self) -> Referee:
        """
        Retrieves the current referee data.

        Returns:
            Referee: The current referee data.
        """
        with self.lock:
            return self.referee

    def get_designated_position(self) -> Optional[Tuple[float, float]]:
        """
        Get the designated position if available.

        Returns:
            Optional[Tuple[float, float]]: The designated position if available, None otherwise.
        """
        if self.referee.designated_position:
            return (
                self.referee.designated_position.x,
                self.referee.designated_position.y,
            )
        return None

    def get_command_counter(self) -> int:
        """
        Get the command counter.

        Returns:
            int: The command counter.
        """
        return self.command_counter

    def check_command_sequence(self, sequence: List[RefereeCommand]) -> bool:
        """
        Check if the last commands match the given sequence.

        Args:
            sequence (List[int]): The sequence of commands to check.

        Returns:
            bool: True if the last commands match the given sequence, False otherwise.
        """
        if len(sequence) > len(self.command_history):
            return False
        return self.command_history[-len(sequence) :] == sequence

    def wait_for_update(self, timeout: float = None) -> bool:
        """
        Waits for the data to be updated, returning True if an update occurs within the timeout.

        Args:
            timeout (float): Maximum time to wait for an update in seconds. Defaults to None (wait indefinitely).

        Returns:
            bool: True if the data was updated within the timeout, False otherwise.
        """
        updated = self.update_event.wait(timeout)
        self.update_event.clear()  # Reset the event for the next update.
        return updated

    def pull_referee_data(self) -> None:
        """
        Continuously receives referee data packets and updates the internal data structures for the game state.

        This method runs indefinitely and should typically be started in a separate thread.
        """
        referee_packet = Referee()
        while True:
            t_received = time.time()
            self.time_received = t_received
            data = self.net.receive_data()
            if data:
                with self.lock:
                    referee_packet.Clear()  # Clear previous data to avoid memory bloat
                    referee_packet.ParseFromString(data)
                    self._update_data(referee_packet)
            self._print_referee_info(t_received, referee_packet)
            time.sleep(0.0083)

    def _print_referee_info(self, t_received: float, referee_packet: Referee) -> None:
        """
        Prints debug information about the referee packet.

        Args:
            t_received (float): The time at which the packet was received.
            referee_packet (Referee): The referee packet containing game state information.
        """
        t_now = time.time()
        logger.debug(f"Time Now          : {t_now:.3f}s")
        logger.debug(f"Referee Command   : {self.string_from_command(self.command)}")
        logger.debug(f"Stage             : {self.string_from_stage(self.stage)}")
        logger.debug(f"Stage Time Left   : {self.stage_time_left} ms")
        logger.debug(f"Command Counter   : {self.command_counter}")
        logger.debug(f"Command Timestamp : {self.command_timestamp} us")
        logger.debug("--- YELLOW TEAM ---------------------------")
        logger.debug(f"{self.yellow_info}")
        logger.debug("--- BLUE TEAM -----------------------------")
        logger.debug(f"{self.blue_info}")
        logger.debug("-------------------------------------------")
