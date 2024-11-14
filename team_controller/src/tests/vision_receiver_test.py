import os
import sys
import time
import threading

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

from team_controller.src.data.vision_receiver import VisionDataReceiver

def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.get_game_data()

def main():
    # Initialize the VisionDataReceiver
    receiver = VisionDataReceiver(debug=False)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    try:
        while True:
            # Wait for the update event with a timeout (optional)
            if receiver.wait_for_update(timeout=1.0):
                # An update has occurred, so process the updated data
                ball_pos = receiver.get_ball_pos()
                robots_yellow_pos = receiver.get_robots_pos(is_yellow=True)
                robots_blue_pos = receiver.get_robots_pos(is_yellow=False)

                print("Updated Ball Position:", ball_pos)
                print("Updated Yellow Robots Positions:", robots_yellow_pos)
                print("Updated Blue Robots Positions:", robots_blue_pos)
            else:
                print("No data update received within the timeout period.")

    except KeyboardInterrupt:
        print("Stopping main program.")

if __name__ == "__main__":
    main()
