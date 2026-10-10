import threading
from collections import deque
from unittest import mock

from utama_core.data_processing.receivers import referee_receiver
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)


def _packet() -> bytes:
    pkt = Referee()
    pkt.packet_timestamp = 1
    pkt.stage = Referee.NORMAL_FIRST_HALF
    pkt.command = Referee.STOP
    pkt.command_counter = 1
    pkt.command_timestamp = 1
    for team in (pkt.yellow, pkt.blue):
        team.name = "x"
        team.score = team.red_cards = team.yellow_cards = team.timeouts = team.timeout_time = team.goalkeeper = 0
    return pkt.SerializeToString()


class _OnePacketNet:
    """Delivers one packet, then blocks the way a socket with no traffic does: a loop that
    spun on `None` instead kept a thread busy for the rest of the test session."""

    def __init__(self):
        self._packets = [_packet()]

    def receive_data(self):
        if self._packets:
            return self._packets.pop()
        threading.Event().wait()


def test_receive_loop_delivers_a_packet():
    # pull_referee_data held a non-reentrant lock while _update_data took it again, so the
    # thread hung on the first game-controller packet and the buffer never filled.
    with mock.patch.object(referee_receiver.network_manager, "NetworkManager", return_value=_OnePacketNet()):
        buffer = deque(maxlen=1)
        receiver = referee_receiver.RefereeMessageReceiver(buffer)

    threading.Thread(target=receiver.pull_referee_data, daemon=True).start()
    for _ in range(200):
        if buffer:
            break
        threading.Event().wait(0.01)

    assert len(buffer) == 1
    assert buffer[0].referee_command == RefereeCommand.STOP
