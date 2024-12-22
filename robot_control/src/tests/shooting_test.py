import os
import sys
import threading

from entities.game import Field

field = Field()

shooter_id = 3
goal_x = -field.half_length
goal_y1 = -field.half_goal_width
goal_y2 = field.half_goal_width

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(project_root)
sys.path.insert(0, project_root)

from team_controller.src.data.vision_receiver import VisionDataReceiver
from robot_control.src.controller.shooting_controller import ShootingController

if __name__ == "__main__":
    vision_receiver = VisionDataReceiver(debug=False)
    decision_maker = ShootingController(
        shooter_id, goal_x, goal_y1, goal_y2, vision_receiver, debug=True
    )

    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    command_thread = threading.Thread(target=decision_maker.approach_ball)

    vision_thread.start()
    command_thread.start()

    try:
        vision_thread.join()
        command_thread.join()
    except KeyboardInterrupt:
        print("Exiting...")
