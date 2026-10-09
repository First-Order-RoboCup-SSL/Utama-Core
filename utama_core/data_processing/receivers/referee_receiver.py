import logging
import time
from typing import Deque

from utama_core.config.settings import MULTICAST_GROUP_REFEREE, REFEREE_PORT
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)
from utama_core.team_controller.src.utils import network_manager

logger = logging.getLogger(__name__)


class RefereeMessageReceiver:
    """A class responsible for receiving and managing referee messages in a multi-robot game environment. The class
    interfaces with a network manager to receive packets, which contain game state information, and updates the internal
    data structures accordingly.

    Args:
        referee_buffer: Where each received message is appended, as `RefereeData`.
        ip (str): The IP address for receiving multicast referee data. Defaults to MULTICAST_GROUP_REFEREE.
        port (int): The port for receiving referee data. Defaults to REFEREE_PORT.
    """

    def __init__(
        self,
        referee_buffer: Deque[RefereeData],
        ip=MULTICAST_GROUP_REFEREE,
        port=REFEREE_PORT,
    ):
        self.net = network_manager.NetworkManager(address=(ip, port), bind_socket=True)
        self.time_received = None
        self.referee_buffer = referee_buffer

        # Initialize state variables
        self.stage = None
        self.command = None
        self.time_sent = None
        self.stage_time_left = None
        self.command_counter = None
        self.command_timestamp = None
        self.yellow_info = TeamInfo("yellow")
        self.blue_info = TeamInfo("blue")

    def _update_data(self, referee_packet: Referee) -> None:
        """Update the internal data structures with the new referee packet.

        Args:
            referee_packet (Referee): The referee packet containing game state information.
        """
        # Update state variables
        self.stage = Stage.from_id(referee_packet.stage)
        self.command = RefereeCommand.from_id(referee_packet.command)
        self.time_sent = referee_packet.packet_timestamp / 1e6  # Convert microseconds to seconds
        self.stage_time_left = referee_packet.stage_time_left / 1e3  # Convert milliseconds to seconds
        self.command_counter = referee_packet.command_counter
        self.command_timestamp = referee_packet.command_timestamp / 1e6  # Convert microseconds to seconds
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
            blue_team=self.blue_info.snapshot(),
            yellow_team=self.yellow_info.snapshot(),
            designated_position=designated_position,
            blue_team_on_positive_half=referee_packet.blue_team_on_positive_half,
            next_command=(
                RefereeCommand.from_id(referee_packet.next_command) if referee_packet.HasField("next_command") else None
            ),
            current_action_time_remaining=(
                referee_packet.current_action_time_remaining
                if referee_packet.HasField("current_action_time_remaining")
                else None
            ),
            game_events=list(referee_packet.game_events),
            match_type=referee_packet.match_type,
            status_message=referee_packet.status_message if referee_packet.status_message else None,
        )

        # deque.append is atomic: the runner's popleft on the main thread needs no lock.
        self.referee_buffer.append(referee_data)

    def pull_referee_data(self) -> None:
        """Continuously receives referee data packets and updates the internal data structures for the game state.

        This method runs indefinitely and should typically be started in a separate thread.
        """
        referee_packet = Referee()
        while True:
            data = self.net.receive_data()
            if data:
                self._handle_packet(referee_packet, data)

    def _handle_packet(self, referee_packet: Referee, data: bytes) -> None:
        self.time_received = time.time()
        referee_packet.Clear()
        referee_packet.ParseFromString(data)
        self._update_data(referee_packet)
