import threading
import queue

from team_controller.src.controllers import GRSimRobotController
from team_controller.src.data import VisionDataReceiver
from team_controller.src.data.message_enum import MessageType
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID

from team_controller.src.config.settings import TIMESTEP


if __name__ == "__main__":
    shooter_id = 3

    game = Game(my_team_is_yellow=True)

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue, debug=False)
    sim_robot_controller = GRSimRobotController(is_team_yellow=True, debug=False)

    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    pid_oren = PID(TIMESTEP, 8, -8, 4.5, 0, 0.045, num_robots=6)
    pid_trans = PID(TIMESTEP, 1.5, -1.5, 4.5, 0, 0.035, num_robots=6)

    try:
        while True:
            (message_type, message) = message_queue.get()
            if message_type == MessageType.VISION:
                game.add_new_state(message)
            elif message_type == MessageType.REF:
                pass

            cmd = score_goal(
                game,
                sim_robot_controller.robot_has_ball(shooter_id),
                shooter_id=shooter_id,
                pid_oren=pid_oren,
                pid_trans=pid_trans,
            )
            sim_robot_controller.add_robot_commands(cmd, shooter_id)
            sim_robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        print("Exiting...")
