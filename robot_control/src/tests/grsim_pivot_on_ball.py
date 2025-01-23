import threading
import queue
import time
import numpy as np
from motion_planning.src.pid.pid import TwoDPID
from team_controller.src.controllers import GRSimRobotController
from team_controller.src.data import VisionDataReceiver
from team_controller.src.data.message_enum import MessageType
from entities.game import Game
from motion_planning.src.pid import PID
from team_controller.src.config.settings import TIMESTEP
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from robot_control.src.skills import turn_on_spot, go_to_ball
import logging

logger = logging.getLogger(__name__)


def test_grsim_pivot_on_ball(shooter_id: int, is_yellow: bool, headless: bool):
    env = GRSimController()
    env.reset()

    game = Game(False)

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue)

    sim_robot_controller = GRSimRobotController(is_team_yellow=False)

    # Start vision thread
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren = PID(TIMESTEP, 8, -8, 10, 0.01, 0.045, num_robots=6)
    # pid_trans = PID(TIMESTEP, 1.5, -1.5, 4.5, 0, 0.035, num_robots=6)
    pid_trans = TwoDPID(TIMESTEP, 2.5, 7.5, 0.01, 0.0, num_robots=6)

    time.sleep(0.1)

    try:
        while True:
            # Process messages from the queue
            if not message_queue.empty():
                (message_type, message) = message_queue.get()

                if message_type == MessageType.VISION:
                    game.add_new_state(message)
                elif message_type == MessageType.REF:
                    pass

            if not sim_robot_controller.robot_has_ball(shooter_id):
                cmd = go_to_ball(
                    pid_oren=pid_oren,
                    pid_trans=pid_trans,
                    this_robot_data=game.get_robot_pos(False, shooter_id),
                    robot_id=shooter_id,
                    ball_data=game.ball,
                )
            else:
                cmd = turn_on_spot(
                    pid_oren=pid_oren,
                    pid_trans=pid_trans,
                    this_robot_data=game.get_robot_pos(False, shooter_id),
                    robot_id=shooter_id,
                    target_oren=-np.pi,
                    dribbling=True,
                    pivot_on_ball=True,
                )

            sim_robot_controller.add_robot_commands(cmd, shooter_id)
            sim_robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        logger.info("Test Interrupted.")
        assert False  # Failure


if __name__ == "__main__":
    # Run the test and output the result
    test_result = test_grsim_pivot_on_ball(5, False, False)
