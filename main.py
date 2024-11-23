import threading
from entities.game import Game
from team_controller.src.data.referee_receiver import RefereeMessageReceiver
from team_controller.src.data.vision_receiver import VisionDataReceiver


def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def main():
    game = Game()
    # Initialize the VisionDataReceiver
    receiver = VisionDataReceiver(debug=True)
    referee_receiver = RefereeMessageReceiver(debug=True)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    referee_thread.daemon = True
    referee_thread.start()

    try:
        while True:
            # Wait for the update event with a timeout (optional)
            if receiver.wait_for_update(timeout=0.1):
                # An update has occurred, so process the updated data
                frame_data = receiver.get_frame_data()
                game.add_new_state(frame_data)

                # access current state data
                # print(
                #     game.current_state.yellow_robots[0].x,
                #     game.current_state.yellow_robots[0].y,
                # )

                # access game records from -x number of frames ago
                # print(game.records[-1].ts, game.records[-1].ball[0].x)
            else:
                print("No data update received within the timeout period.")

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
