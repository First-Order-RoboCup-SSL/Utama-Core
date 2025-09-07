import logging
import queue
import threading

from utama_core.entities.game import Game
from utama_core.team_controller.src.data import VisionReceiver
from utama_core.team_controller.src.data.message_enum import MessageType

logger = logging.getLogger(__name__)
import warnings

# Emit a deprecation notice indicating upcoming relocation of this utility.
warnings.warn(
    "real_vision_test.py will be moved out of team_controller soon. "
    "Prefer using `utama_core.run.receivers.vision_receiver.VisionReceiver`.",
    DeprecationWarning,
    stacklevel=2,
)


def data_update_listener(receiver: VisionReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def main():
    game = Game()

    message_queue = queue.SimpleQueue()
    receiver = VisionReceiver(message_queue)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    # TODO: Not implemented
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    # referee_thread.daemon = True
    # referee_thread.start()

    try:
        while True:
            (message_type, message) = message_queue.get()  # Infinite timeout for now

            if message_type == MessageType.VISION:
                game.add_new_state(message)
                print(message)
            elif message_type == MessageType.REF:
                pass

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
