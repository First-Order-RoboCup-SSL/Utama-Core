import threading

from team_controller.src.data.vision_receiver import VisionDataReceiver
from team_controller.src.controllers.robot_startup_controller import StartUpController
from entities.game import Game


# example of accessing data from vision
def access_vision_data(vision_receiver: VisionDataReceiver, game: Game):
    while True:
        robots = vision_receiver.get_robots_pos(is_yellow=True)
        robots_blue = vision_receiver.get_robots_pos(is_yellow=False)
        balls = vision_receiver.get_ball_pos()
        print(robots_blue, robots)
        print(balls)


if __name__ == "__main__":
    game = Game()
    vision_receiver = VisionDataReceiver(debug=False)
    # decision_maker = StartUpController(vision_receiver, debug=False)

    vision_thread = threading.Thread(target=vision_receiver.get_game_data)
    access_thread = threading.Thread(
        target=access_vision_data,
        args=(
            vision_receiver,
            game,
        ),
    )
    # command_thread = threading.Thread(target=decision_maker.startup)

    vision_thread.start()
    access_thread.start()
    # command_thread.start()

    try:
        vision_thread.join()
        # robots = vision_thread.get_robot_dict(is_yellow=True)
        access_thread.join()
        # command_thread.join()
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(e)
