from typing import Tuple, Optional

from team_controller.src.utils import network_manager
from team_controller.src.config.settings import MULTICAST_GROUP_REFEREE, REFEREE_PORT

from team_controller.src.generated_code.ssl_gc_referee_message_pb2 import Referee


class RefereeMessageReceiver:
    def __init__(self, ip=MULTICAST_GROUP_REFEREE, port=REFEREE_PORT):
        self.net = network_manager.NetworkManager(address=(ip, port), bind_socket=True)

        self.prev_command_counter = -1
        self.command_history = []
        self.latest_message = Referee()  # Initialize a single Referee object
        self.old_serialized_data = None

    def _serialize_relevant_fields(self, data: bytes) -> bytes:
        # Create a shallow copy of the message
        message_copy = Referee()
        message_copy.ParseFromString(data)

        # Exclude `packet_timestamp` and `stage_time_left` by setting them to a default value
        message_copy.packet_timestamp = 0
        message_copy.stage_time_left = 0

        # Serialize the message to a byte string
        return message_copy

    def check_new_message(self) -> bool:
        data = self.net.receive_data()
        if data:
            serialized_data = self._serialize_relevant_fields(data)

        if serialized_data != self.old_serialized_data:
            self.latest_message.ParseFromString(data)
            return True
        return False

    def check_new_command(self) -> bool:
        data = self.net.receive_data()
        history_length = 5

        if data:
            self.latest_message.ParseFromString(data)  # Reuse the same object
            if self.latest_message.command_counter != self.prev_command_counter:
                self.prev_command_counter = self.latest_message.command_counter
                self.command_history.append(self.latest_message.command)
                if len(self.command_history) > history_length:
                    self.command_history.pop(0)  # Maintain a fixed-length history
                print(self.command_history)
                return True
        return False

    def get_latest_command(self):
        command = self.latest_message.command
        # print(Referee.Command.Name(command))
        designated_position = self.latest_message.designated_position
        return command, (designated_position.x, designated_position.y)

    def get_latest_message(self) -> object:
        return self.latest_message

    def get_stage_time_left(self) -> float:
        return self.latest_message.stage_time_left / 1000000

    def get_packet_timestamp(self) -> float:
        return self.latest_message.packet_timestamp / 1000000

    def yellow_team_info(self) -> object:
        return self.latest_message.yellow

    def blue_team_info(self) -> object:
        return self.latest_message.blue

    def get_next_command(self) -> Optional[int]:
        if self.latest_message.next_command:
            return self.latest_message.next_command
        return None

    def get_designated_position(self) -> Optional[Tuple[float, float]]:
        if self.latest_message.designated_position:
            return (
                self.latest_message.designated_position.x,
                self.latest_message.designated_position.y,
            )
        return None

    def get_command_counter(self) -> int:
        return self.latest_message.command_counter

    def check_command_sequence(self, sequence: list) -> bool:
        """Check if the last commands match the given sequence."""
        if len(sequence) > len(self.command_history):
            return False
        return self.command_history[-len(sequence) :] == sequence
