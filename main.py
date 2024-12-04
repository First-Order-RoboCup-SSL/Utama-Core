import threading
import queue
from entities.game import Game
import time
import math

from team_controller.src.controllers.sim.grsim_controller import GRSimController
from team_controller.src.controllers.sim.robot_startup_controller import (
    StartUpController,
)
from team_controller.src.data import VisionDataReceiver, RefereeMessageReceiver
from team_controller.src.data.message_enum import MessageType


def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def main():
    game = Game()
    GRSimController().teleport_ball(0, 0, 2, 2.5)
    time.sleep(0.2)

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

    frames = 0

    try:
        print("LOCATED BALL")
        print("Predicting ball position with 0.5 seconds of motion")

        predictions = []
        while True:
            (message_type, message) = message_queue.get()  # Infinite timeout for now
            
            if message_type == MessageType.VISION:
                frames += 1

                if frames % 10 == 0:
                    predictions.append(game.predict_ball_pos_after(0.5))
                    actual = game.get_ball_pos()
                    if (len(predictions)) >= 4 and predictions[-4] != None:
                      print("Prediction inaccuracy delta (cm): ", 100 * math.sqrt((actual[0].x - predictions[-4][0])**2 + (actual[0].y - predictions[-4][1])**2))

                game.add_new_state(message)

            elif message_type == MessageType.REF:
                pass

            decision_maker.make_decision()

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
