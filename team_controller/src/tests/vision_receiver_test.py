import threading
import logging

logger = logging.getLogger(__name__)

# TODO: Tests need to be updated.

from team_controller.src.data import VisionReceiver


def data_update_listener(receiver: VisionReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def main():
    # Initialize the VisionDataReceiver
    receiver = VisionReceiver()

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    try:
        while True:
            # Wait for the update event with a timeout (optional)
            if receiver.wait_for_update(timeout=1.0):
                # An update has occurred, so process the updated data
                ball_pos = receiver.get_ball_pos()  # TEST TODO
                robots_yellow_pos = receiver.get_robots_pos(is_yellow=True)  # TESTTODO
                robots_blue_pos = receiver.get_robots_pos(is_yellow=False)  # TESTTODO
                robot_coords = receiver.get_robot_coords(is_yellow=False)  # TESTTODO

                logger.info(f"Updated Ball Position: {ball_pos}")
                logger.info(f"Updated Yellow Robots Positions: {robots_yellow_pos}")
                logger.info(f"Updated Blue Robots Positions: {robots_blue_pos}")
                logger.info(f"Update Blue Robots Coords: {robot_coords}")
            else:
                logger.warning("No data update received within the timeout period.")

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
