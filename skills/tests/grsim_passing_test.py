import threading
import queue
import time
from team_controller.src.controllers import GRSimRobotController
from team_controller.src.data import VisionReceiver
from team_controller.src.data.message_enum import MessageType
from entities.game import Game
from robot_control.src.intent import PassBall
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from motion_planning.src.pid.pid import get_grsim_pids
import logging

logger = logging.getLogger(__name__)

TARGET_COORDS = (-2, 3)


def test_grsim_passing(
    passer_id: int, receiver_id: int, is_yellow: bool, headless: bool
):
    env = GRSimController()
    env.reset()

    game = Game(False)

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionReceiver(message_queue)

    sim_robot_controller = GRSimRobotController(is_team_yellow=False)

    # Start vision thread
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren, pid_trans = get_grsim_pids(6)

    time.sleep(0.3)

    passed = False

    pass_ball_task = PassBall(
        pid_oren,
        pid_trans,
        game,
        passer_id,
        receiver_id,
        target_coords=TARGET_COORDS,
    )

    try:
        while True:
            # Process messages from the queue
            if not message_queue.empty():
                (message_type, message) = message_queue.get()

                if message_type == MessageType.VISION:
                    game.add_new_state(message)
                elif message_type == MessageType.REF:
                    pass

            if not passed:
                passer_cmd, receiver_cmd = pass_ball_task.enact(
                    passer_has_ball=sim_robot_controller.robot_has_ball(passer_id)
                )

                if sim_robot_controller.robot_has_ball(receiver_id):
                    logger.info("Passed.")
                    passed = True
                    time.sleep(1)
                    break

                sim_robot_controller.add_robot_commands(passer_cmd, passer_id)
                sim_robot_controller.add_robot_commands(receiver_cmd, receiver_id)
                sim_robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        logger.info("Test Interrupted.")
        assert False  # Failure


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # Run the test and output the result
    test_result = test_grsim_passing(4, 5, False, False)
