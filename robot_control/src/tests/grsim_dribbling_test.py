import logging
import random
import threading
import queue
from team_controller.src.controllers.sim.grsim_controller import GRSimController
from team_controller.src.controllers.sim.grsim_robot_controller import (
    GRSimRobotController,
)
from config.settings import TIMESTEP
from motion_planning.src.pid.pid import get_grsim_pids
from team_controller.src.data import VisionReceiver
from team_controller.src.data.message_enum import MessageType
from robot_control.src.high_level_skills import DribbleToTarget
from rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from entities.game import Game


def test_grsim_dribbling(dribbler_id: int, is_yellow: bool, headless: bool):
    game = Game()

    message_queue = queue.SimpleQueue()
    vision_receiver = VisionReceiver(message_queue)
    robot_controller = GRSimRobotController(is_team_yellow=True)

    vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
    vision_thread.daemon = True
    vision_thread.start()

    pid_oren, pid_trans = get_grsim_pids(6)

    target_coords = [(4, 2.5), (4, -2.5), (-4, -2.5), (-4, 2.5)]
    idx = 0
    dribble_task = DribbleToTarget(
        pid_oren,
        pid_trans,
        game,
        dribbler_id,
        target_coords=target_coords[0],
        augment=True,
    )

    try:
        while True:
            (message_type, message) = message_queue.get()
            if message_type == MessageType.VISION:
                game.add_new_state(message)
            elif message_type == MessageType.REF:
                pass

            f, e, b = game.get_my_latest_frame(my_team_is_yellow=is_yellow)

            if (
                (f[dribbler_id].x - target_coords[idx][0]) ** 2
                + (f[dribbler_id].y - target_coords[idx][1]) ** 2
            ) < 0.1:
                idx = (idx + 1) % 4
                dribble_task.update_coord(target_coords[idx])
                print("SUCCESS")

            cmd = dribble_task.enact(robot_controller.robot_has_ball(dribbler_id))
            if dribble_task.dribbled_distance > 1.0:
                print("FOULLL: ", dribble_task.dribbled_distance)
            robot_controller.add_robot_commands(cmd, dribbler_id)
            robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # Run the test and output the result
    test_result = test_grsim_dribbling(3, False, False)
