import threading
import queue
from entities.game import Game
import time

from team_controller.src.controllers.robot_startup_controller import StartUpController
from team_controller.src.controllers.sim_controller import SimulatorController
from team_controller.src.data import VisionDataReceiver, RefereeMessageReceiver
from team_controller.src.data.message_enum import MessageType


def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def main():
    game = Game()
    SimulatorController().teleport_ball(0, 0, 2, 2.5)
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

    start = time.time()
    frames = 0

    try:
        print("LOCATED BALL")
        predictions = []
        while True:
            (message_type, message) = message_queue.get()  # Infinite timeout for now
            
            if message_type == MessageType.VISION:
                frames += 1

                if frames % 10 == 0:
                    # print((message_queue.qsize() + frames) / (time.time() - start))
                    predictions.append(game.predict_ball_pos_after(0.5))
                    print("POS", game.get_ball_pos(), time.time())
                    if (len(predictions)) >= 4:
                      print("PRED_NOW", predictions[-4])

                    # print(message_queue.qsize())
                # message = FrameData(...)
                game.add_new_state(message)
                # access current state data
                # print(
                #     game.current_state.yellow_robots[0].x,
                #     game.current_state.yellow_robots[0].y,
                # )

                # access game records from -x number of frames ago
                # print(game.records[-1].ts, game.records[-1].ball[0].x)

            elif message_type == MessageType.REF:
                pass

            decision_maker.make_decision()

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
