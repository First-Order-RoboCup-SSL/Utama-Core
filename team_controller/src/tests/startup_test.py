import os
import sys
import threading
import queue
from entities.game import Game
import time

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(project_root)
sys.path.insert(0, project_root)

from team_controller.src.data.message_enum import MessageType
from team_controller.src.data import VisionDataReceiver
from team_controller.src.controllers.sim.robot_startup_controller import (
    StartUpController,
)


def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()

def main():
    game = Game()

    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, debug=False)
    decision_maker = StartUpController(game, debug=False)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    # TODO: Not implemented
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    # referee_thread.daemon = True
    # referee_thread.start()

    start = time.time()
    frames = 0

    try:
        while True:
            (message_type, message) = message_queue.get()  # Infinite timeout for now
            
            if message_type == MessageType.VISION:
                game.add_new_state(message)
            elif message_type == MessageType.REF:
                pass

            decision_maker.make_decision()

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()

