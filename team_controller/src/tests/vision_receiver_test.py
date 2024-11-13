import os
import sys
import time

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

from team_controller.src.data.vision_receiver import VisionDataReceiver

if __name__ == "__main__":
    # Initialize the VisionDataReceiver
    vision_receiver = VisionDataReceiver(debug=True)

    # Start receiving and processing vision data in a separate thread
    import threading

    threading.Thread(target=vision_receiver.get_game_data).start()

    # Retrieve and print the current positions of robots and the ball
    while True:
        yellow_robots = vision_receiver.get_robots_pos(is_yellow=True)
        blue_robots = vision_receiver.get_robots_pos(is_yellow=False)
        ball_position = vision_receiver.get_ball_pos()

        print(f"Yellow Team Robots: {yellow_robots}")
        print(f"Blue Team Robots: {blue_robots}")
        print(f"Ball Position: {ball_position}")

        time.sleep(1)
