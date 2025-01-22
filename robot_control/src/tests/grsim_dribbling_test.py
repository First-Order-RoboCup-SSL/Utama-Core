import random
import threading
import queue
from team_controller.src.controllers.sim.grsim_robot_controller import (
    GRSimRobotController,
)
from team_controller.src.config.settings import TIMESTEP
from motion_planning.src.pid.pid import get_pids
from team_controller.src.data import VisionDataReceiver
from team_controller.src.data.message_enum import MessageType
from robot_control.src.high_level_skills import DribbleToTarget
from rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from entities.game import Game

IS_YELLOW = True

robot_id = 3
target_coords = (-2, 1)

game = Game()

message_queue = queue.SimpleQueue()
vision_receiver = VisionDataReceiver(message_queue)
robot_controller = GRSimRobotController(is_team_yellow=True)

vision_thread = threading.Thread(target=vision_receiver.pull_game_data)
vision_thread.daemon = True
vision_thread.start()

pid_oren, pid_trans = get_pids(6)

target_coords = (random.random() * 4 - 2, random.random() * 4 - 2)
dribble_task = DribbleToTarget(
    pid_oren,
    pid_trans,
    game,
    robot_id,
    target_coords=target_coords,
)
my_team_is_yellow = True

try:
    while True:
        (message_type, message) = message_queue.get()
        if message_type == MessageType.VISION:
            game.add_new_state(message)
        elif message_type == MessageType.REF:
            pass

        f, e, b = game.get_my_latest_frame(my_team_is_yellow=my_team_is_yellow)

        if (
            (f[robot_id].x - target_coords[0]) ** 2
            + (f[robot_id].y - target_coords[1]) ** 2
        ) < 0.1:
            print("SUCCESS")
            target_coords = (random.random() * 4 - 2, random.random() * 4 - 2)
            dribble_task.update_coord(target_coords)

        cmd = dribble_task.enact(robot_controller.robot_has_ball(robot_id))
        if dribble_task.dribbled_distance > 1.2:
            print("FOULLL")
        robot_controller.add_robot_commands(cmd, robot_id)
        robot_controller.send_robot_commands()

except KeyboardInterrupt:
    print("Exiting...")
