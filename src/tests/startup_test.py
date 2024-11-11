import os
import sys
import threading

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(project_root)
sys.path.insert(0, project_root)

from data.vision_receiver import VisionDataReceiver
from controllers.robot_startup_controller import StartUpController

if __name__ == "__main__":
    vision_receiver = VisionDataReceiver(debug=True)
    decision_maker = StartUpController(vision_receiver, debug=False)

    vision_thread = threading.Thread(target=vision_receiver.get_game_data)
    command_thread = threading.Thread(target=decision_maker.startup)

    vision_thread.start()
    command_thread.start()

    try:
        vision_thread.join()
        command_thread.join()
    except KeyboardInterrupt:
        print("Exiting...")