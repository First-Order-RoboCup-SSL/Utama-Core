import threading
import queue
from motion_planning.src.pid.pid import TwoDPID
from robot_control.src.skills import turn_on_spot
from team_controller.src.controllers import GRSimRobotController
from team_controller.src.data import VisionDataReceiver
from team_controller.src.data.message_enum import MessageType
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID

from team_controller.src.config.settings import TIMESTEP


if __name__ == "__main__":
    IS_YELLOW = True

    shooter_id = 3

    game = Game()

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue)
    sim_robot_controller = GRSimRobotController(is_team_yellow=True)

    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    pid_oren = PID(TIMESTEP, 8, -8, 4.5, 0, 0.045, num_robots=6)
    # pid_trans = PID(TIMESTEP, 1.5, -1.5, 4.5, 0, 0.035, num_robots=6)
    pid_trans = TwoDPID(TIMESTEP, 1.5, -1.5, 3, 0.1, 0.0, num_robots=6)

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
                is_yellow=IS_YELLOW
            )
            # cmd = turn_on_spot(pid_oren, pid_trans, )
            sim_robot_controller.add_robot_commands(cmd, shooter_id)
            sim_robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        print("Exiting...")
