import threading
import queue
import time
from motion_planning.src.pid.pid import TwoDPID
from team_controller.src.controllers import GRSimRobotController
from team_controller.src.data import VisionDataReceiver
from team_controller.src.data.message_enum import MessageType
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.config.settings import TIMESTEP
from team_controller.src.controllers.sim.grsim_controller import GRSimController
import logging
import random

logger = logging.getLogger(__name__)

# Test parameters
MAX_TIME = 30  # Maximum time (in seconds) to score a goal


def test_grsim_shooting(shooter_id: int, is_yellow: bool, headless: bool):
    """Run the shooting test and return whether the robot scored within the time limit."""
    env = GRSimController()
    env.reset()

    game = Game()

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionDataReceiver(message_queue)

    sim_robot_controller = GRSimRobotController(is_team_yellow=is_yellow)

    # Start vision thread
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    # Initialize PID controllers
    pid_oren = PID(TIMESTEP, 8, -8, 10, 0.01, 0.045, num_robots=6)
    # pid_trans = PID(TIMESTEP, 1.5, -1.5, 4.5, 0, 0.035, num_robots=6)
    pid_trans = TwoDPID(TIMESTEP, 2.5, 7.5, 0.01, 0.0, num_robots=6)

    time.sleep(0.1)

    start_time = time.time()  # Start the timer
    shoot_in_left_goal = random.random() > 0.5

    try:
        while True:
            # Check if the time limit has been exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_TIME:
                logger.info("Test Failed: Time limit exceeded.")
                assert False  # Failure

            # Process messages from the queue
            if not message_queue.empty():
                (message_type, message) = message_queue.get()

                if message_type == MessageType.VISION:
                    game.add_new_state(message)
                elif message_type == MessageType.REF:
                    pass

            # Generate commands for the shooter robot
            cmd = score_goal(
                game,
                sim_robot_controller.robot_has_ball(shooter_id),
                shooter_id=shooter_id,
                pid_oren=pid_oren,
                pid_trans=pid_trans,
                is_yellow=is_yellow,
                shoot_in_left_goal=shoot_in_left_goal,
            )
            sim_robot_controller.add_robot_commands(cmd, shooter_id)
            sim_robot_controller.send_robot_commands()

            # Check if a goal has been scored
            if game.is_ball_in_goal(right_goal=not is_yellow):
                logger.info(f"Test Passed: Goal scored in {elapsed_time:.2f} seconds.")
                break

        assert True  # Success

    except KeyboardInterrupt:
        logger.info("Test Interrupted.")
        assert False  # Failure


if __name__ == "__main__":
    # Run the test and output the result
    test_result = test_grsim_shooting(5, False, False)
