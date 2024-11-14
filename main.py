import threading
from entities.game import Game
from team_controller.src.data.vision_receiver import VisionDataReceiver


def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.get_game_data()


def main():
    game = Game()
    # Initialize the VisionDataReceiver
    receiver = VisionDataReceiver(debug=False)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    try:
        while True:
            # Wait for the update event with a timeout (optional)
            if receiver.wait_for_update(timeout=0.1):
                # An update has occurred, so process the updated data
                ball_pos = receiver.get_ball_pos()
                robots_yellow_pos = receiver.get_robots_pos(is_yellow=True)
                robots_blue_pos = receiver.get_robots_pos(is_yellow=False)
                time_received = receiver.get_time_received()
                game.add_state_from_vision(
                    time_received, robots_yellow_pos, robots_blue_pos, ball_pos
                )
                print(game.current_state.robots["yellow"][0].x)
            else:
                print("No data update received within the timeout period.")

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
