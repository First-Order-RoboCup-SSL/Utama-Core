import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(project_root)
sys.path.insert(0, project_root)

from data.vision_receiver import VisionDataReceiver

if __name__ == "__main__":
    game_data = VisionDataReceiver(debug=True)
    try:
        game_data.get_game_data()
    except KeyboardInterrupt:
        print("Exiting...")