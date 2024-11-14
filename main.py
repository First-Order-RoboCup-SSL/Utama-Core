import threading

from team_controller.src.data.vision_receiver import VisionDataReceiver
from team_controller.src.controllers.robot_startup_controller import StartUpController

if __name__ == "__main__":
    vision_receiver = VisionDataReceiver(debug=True)
    decision_maker = StartUpController(vision_receiver, debug=False)

    vision_thread = threading.Thread(target=vision_receiver.get_game_data)
    # command_thread = threading.Thread(target=decision_maker.startup)

    vision_thread.start()
    # command_thread.start()

    try:
        vision_thread.join()
        robots = vision_thread.get_robot_dict(is_yellow=True)
        balls = vision_thread.get_ball_dict()
        print(balls)
        # command_thread.join()
    except KeyboardInterrupt:
        print("Exiting...")
