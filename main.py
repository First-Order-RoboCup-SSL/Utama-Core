import threading
import queue
from entities.game import Game
import time
import math
from typing import List

from entities.data.vision import FrameData
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from team_controller.src.tests.grsim_robot_controller_startup_test import (
    StartUpController,
)
from team_controller.src.data import VisionDataReceiver, RefereeMessageReceiver
from team_controller.src.data.message_enum import MessageType
import logging

logger = logging.getLogger(__name__)

def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()


def main():
    game = Game()
    GRSimController().teleport_ball(0, 0, 2, 2.5)
    time.sleep(0.2)

    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue, debug=False)
    decision_maker = StartUpController(game, debug=False)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    TIME = 0.5
    FRAMES_IN_TIME = round(60 * TIME)

    # TODO: Not implemented
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    # referee_thread.daemon = True
    # referee_thread.start()

    frames = 0

    try:
        logger.debug("LOCATED BALL")
        logger.debug(f"Predicting robot position with {TIME} seconds of motion")

        predictions: List[FrameData] = []
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
                        "Ball prediction inaccuracy delta (cm): ",
                        "{:.20f}".format(
                            100
                            * math.sqrt(
                                (
                                    actual.ball[0].x
                                    - predictions[-FRAMES_IN_TIME].ball[0].x
                                )
                                ** 2
                                + (
                                    actual.ball[0].y
                                    - predictions[-FRAMES_IN_TIME].ball[0].y
                                )
                                ** 2
                            )
                        ),
                    )
                    for i in range(6):
                        logger.debug(
                            f"Blue robot {i} prediction inaccuracy delta (cm): ",
                            "{:.20f}".format(
                                100
                                * math.sqrt(
                                    (
                                        actual.blue_robots[i].x
                                        - predictions[-FRAMES_IN_TIME].blue_robots[i].x
                                    )
                                    ** 2
                                    + (
                                        actual.blue_robots[i].y
                                        - predictions[-FRAMES_IN_TIME].blue_robots[i].y
                                    )
                                    ** 2
                                )
                            ),
                        )
                    for i in range(6):
                        logger.debug(
                            f"Yellow robot {i} prediction inaccuracy delta (cm): ",
                            "{:.20f}".format(
                                100
                                * math.sqrt(
                                    (
                                        actual.yellow_robots[i].x
                                        - predictions[-FRAMES_IN_TIME]
                                        .yellow_robots[i]
                                        .x
                                    )
                                    ** 2
                                    + (
                                        actual.yellow_robots[i].y
                                        - predictions[-FRAMES_IN_TIME]
                                        .yellow_robots[i]
                                        .y
                                    )
                                    ** 2
                                )
                            ),
                        )

                predictions.append(game.predict_frame_after(TIME))

            elif message_type == MessageType.REF:
                pass

            decision_maker.make_decision()

    except KeyboardInterrupt:
        logger.info("Stopping main program.")


if __name__ == "__main__":
    main()
