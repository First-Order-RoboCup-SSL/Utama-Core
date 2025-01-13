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

dribble_task = DribbleToTarget(
    pid_oren,
    pid_trans,
    game,
    robot_id,
    target_coords=target_coords,
)

try:
    while True:
        (message_type, message) = message_queue.get()
        if message_type == MessageType.VISION:
            game.add_new_state(message)
        elif message_type == MessageType.REF:
            pass

        cmd = dribble_task.enact(robot_controller.robot_has_ball(robot_id))
        # cmd = turn_on_spot(pid_oren, pid_trans, )
        robot_controller.add_robot_commands(cmd, robot_id)
        robot_controller.send_robot_commands()

except KeyboardInterrupt:
    print("Exiting...")
