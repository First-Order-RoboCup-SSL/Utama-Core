import threading
import queue
from entities.game import Game
import time
import math
from typing import List, Type


from entities.data.command import RobotResponse
from entities.data.vision import FrameData
from vision.vision_processor import VisionProcessor
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from team_controller.src.tests.grsim_robot_controller_startup_test import (
    StartUpController,
)

from team_controller.src.data import VisionReceiver, RefereeMessageReceiver
from team_controller.src.data.message_enum import MessageType
import logging

logger = logging.getLogger(__name__)

import warnings

# Enable all warnings, including DeprecationWarning
warnings.simplefilter("default", DeprecationWarning)


def data_update_listener(receiver: VisionReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def main():
    game = Game(my_team_is_yellow=True, num_friendly_robots=6, num_enemy_robots=6)
    GRSimController().teleport_ball(0, 0, 2, 2.5)
    time.sleep(0.2)

    message_queue = queue.SimpleQueue()

    referee_receiver = RefereeMessageReceiver(message_queue, debug=False)
    vision_receiver = VisionReceiver(message_queue)
    decision_maker = StartUpController(game)

    # Start the data receiving in separate threads
    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)

    # Allows the thread to close when the main program exits
    vision_thread.daemon = True
    referee_thread.daemon = True

    # Start both thread
    vision_thread.start()
    referee_thread.start()

    TIME = 1 / 60 * 10  # frames in seconds
    FRAMES_IN_TIME = round(60 * TIME)

    # TODO: Not implemented
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    # referee_thread.daemon = True
    # referee_thread.start()
    frames = 0

    # To debug
    logging.basicConfig(level=logging.DEBUG)

    try:
        logger.debug(
            f"Predicting robot position with {FRAMES_IN_TIME} frames of motion"
        )

        predictions: List[FrameData] = []

        startup_waiter = VisionProcessor()
        while not startup_waiter.is_ready():
            (message_type, message) = message_queue.get()  # Infinite timeout for now
            if message_type == MessageType.VISION:
                predictions.add_new_State(message)

        game = predictions.get_game()

        while True:
            (message_type, message) = message_queue.get()  # Infinite timeout for now
            if message_type == MessageType.VISION:
                frames += 1
                game.add_new_state(message)

                actual = game.records[-1]  # JUST FOR TESTING - don't do this irl
                if (
                    len(predictions) >= FRAMES_IN_TIME
                    and predictions[-FRAMES_IN_TIME] != None
                ):
                    logger.debug(
                        "Ball prediction inaccuracy delta (cm): "
                        + "{:.5f}".format(
                            100
                            * math.sqrt(
                                (game.ball.x - predictions[-FRAMES_IN_TIME].ball[0].x)
                                ** 2
                                + (game.ball.y - predictions[-FRAMES_IN_TIME].ball[0].y)
                                ** 2
                            )
                        )
                    )
                    for i in range(6):
                        logger.debug(
                            f"Enemy(Blue) robot {i} prediction inaccuracy delta (cm): "
                            + "{:.5f}".format(
                                100
                                * math.sqrt(
                                    (
                                        # proposed implementation
                                        game.enemy_robots[i].x
                                        - predictions[-FRAMES_IN_TIME].enemy_robots[i].x
                                    )
                                    ** 2
                                    + (
                                        # reordered frame implementation
                                        game.enemy_robots[i].y
                                        - predictions[-FRAMES_IN_TIME].enemy_robots[i].y
                                    )
                                    ** 2
                                )
                            )
                        )
                    for i in range(6):
                        logger.debug(
                            f"Friendly(Yellow) robot {i} prediction inaccuracy delta (cm): "
                            + "{:.5f}".format(
                                100
                                * math.sqrt(
                                    (
                                        # original implementation
                                        game.friendly_robots[i].x
                                        - predictions[-FRAMES_IN_TIME]
                                        .friendly_robots[i]
                                        .x
                                    )
                                    ** 2
                                    + (
                                        # proposed implementation
                                        game.friendly_robots[i].y
                                        - predictions[-FRAMES_IN_TIME]
                                        .friendly_robots[i]
                                        .y
                                    )
                                    ** 2
                                )
                            )
                        )

                predictions.append(game.predicted_next_frame)

            elif message_type == MessageType.REF:
                game.add_new_referee_data(message)

            decision_maker.make_decision()

    except KeyboardInterrupt:
        print("Stopping main program.")


def main1():
    """
    This is a test function to demonstrate the use of the Game class and the Robot class.

    In terms of RobotInfo and the way it is currently implemented is not confirmed and may change.
    (have the robot info update game directly in the main thread as robot commands are sent on the main thread)

    We will need to implement a way to update the robot info with grsim and rsim in the controllers
    """
    ### Standard setup for the game ###
    game = Game(my_team_is_yellow=True)
    time.sleep(0.2)

    message_queue = queue.SimpleQueue()
    receiver = VisionReceiver(message_queue)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    # TODO: Not implemented
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    # referee_thread.daemon = True
    # referee_thread.start()

    #### Demo ####

    ### Creates the made up robot info message ###
    madeup_recieved_message = [
        RobotResponse(True),
        RobotResponse(False),
        RobotResponse(False),
        RobotResponse(False),
        RobotResponse(False),
        RobotResponse(False),
    ]
    message_type = MessageType.ROBOT_INFO
    message_queue.put((message_type, madeup_recieved_message))

    try:
        while True:
            (message_type, message) = message_queue.get()

            if message_type == MessageType.VISION:
                game.add_new_state(message)

                ### for demo purposes (displays when vision is received) ###
                print(
                    f"Before robot is_active( {game.friendly_robots[0].inactive} ) coords: {game.friendly_robots[0].x}, {game.friendly_robots[0].y}"
                )
                # TODO: create a check with referee to see if robot is inactive
                game.friendly_robots[0].inactive = True
                print(
                    f"After robot is_active( {game.friendly_robots[0].inactive} ) Coords: {game.friendly_robots[0].x}, {game.friendly_robots[0].y}\n"
                )

                ### Getting coordinate data ###
                print(
                    f"Friendly(Yellow) Robot 1 coords: {game.friendly_robots[0].x}, {game.friendly_robots[0].y}, {game.friendly_robots[0].orientation}"
                )
                print(f"Ball coords: {game.ball.x}, {game.ball.y}, {game.ball.z}\n\n")

            if message_type == MessageType.REF:
                pass

            if message_type == MessageType.ROBOT_INFO:
                game.add_robot_info(message)

                ### for demo purposes (displays when robot info is received) ####
                for i in range(6):
                    print(f"Robot {i} has ball: {game.friendly_robots[i].has_ball}\n\n")

    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    main()
