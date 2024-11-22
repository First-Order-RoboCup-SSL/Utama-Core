import threading
import queue
from entities.game import Game
from team_controller.src.controllers.robot_startup_controller import StartUpController
from team_controller.src.data.vision_receiver import VisionDataReceiver
from team_controller.src.data.message_enum import MessageType



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

    try:
        while True:
            (message_type, message) = message_queue.get() # Infinite timeout for now

            if message_type == MessageType.VISION:
                # message = FrameData(...)
                game.add_new_state(message)
                # access current state data
                print(
                    game.current_state.yellow_robots[0].x,
                    game.current_state.yellow_robots[0].y,
                )

                # access game records from -x number of frames ago
                print(game.records[-1].ts, game.records[-1].ball[0].x)

            elif message_type == MessageType.REF:
                pass
        
            decision_maker.make_decision()

    except KeyboardInterrupt:
        print("Stopping main program.")

if __name__ == "__main__":
    main()
