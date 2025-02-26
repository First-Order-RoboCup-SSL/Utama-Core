import threading
import queue
from entities.game import Game
import time
from team_controller.src.data.message_enum import MessageType
from team_controller.src.data import VisionDataReceiver

import time
import numpy as np
from typing import Tuple, Optional, Dict, Union, List
from global_utils.math_utils import rotate_vector
from entities.data.command import RobotVelCommand
from entities.data.vision import RobotData, BallData
from team_controller.src.controllers import GRSimRobotController
from motion_planning.src.pid.pid import PID
from config.settings import (
    PID_PARAMS,
)
from config.starting_formation import YELLOW_START_ONE
from team_controller.src.generated_code.ssl_simulation_robot_control_pb2 import (
    RobotControl,
)
import logging

logger = logging.getLogger(__name__)
sim_robot_controller = GRSimRobotController(is_team_yellow=True)

def inverse_kinematics(forward_vel: float, left_vel:float, angular_vel: float, ) -> List[float]:
    """
    Calculate the wheel velocities for a robot given the desired forward, left, and angular velocities.
    """
    # Example usage
    wheel_angles = [np.deg2rad(57.3), np.deg2rad(137.03), np.deg2rad(222.97), np.deg2rad(302.7)]  # Wheel angles in radians

    r = 0.178  # Distance from center to wheel in meters
    
    wheel_velocities = []
    for theta in wheel_angles:
        v_i = forward_vel * np.sin(theta) + left_vel * np.cos(theta) + angular_vel * r
        wheel_velocities.append(v_i)
    return wheel_velocities

def packet_builder():
    out_packet = RobotControl()
    # idk why its soo slow but if its to fast something breaks
    wheel_velocities = inverse_kinematics(forward_vel=0, left_vel=10, angular_vel=0)
    command = RobotVelCommand(
            front_right=wheel_velocities[0],
            back_right=wheel_velocities[1],
            back_left=wheel_velocities[2],
            front_left=wheel_velocities[3],
            kick=0,
            chip=0,
            dribble=0,
        )
    sim_robot_controller._add_robot_wheel_vel_command(command, 0)
    logger.debug(out_packet)
    sim_robot_controller.send_robot_commands()

def data_update_listener(receiver: VisionDataReceiver):
    # Start receiving game data; this will run in a separate thread.
    receiver.pull_game_data()

def main():
    game = Game()

    message_queue = queue.SimpleQueue()
    receiver = VisionDataReceiver(message_queue)

    # Start the data receiving in a separate thread
    data_thread = threading.Thread(target=data_update_listener, args=(receiver,))
    data_thread.daemon = True  # Allows the thread to close when the main program exits
    data_thread.start()

    # TODO: Not implemented
    # referee_thread = threading.Thread(target=referee_receiver.pull_referee_data)
    # referee_thread.daemon = True
    # referee_thread.start()

    start = time.time()
    frames = 0

    try:
        while True:
            (message_type, message) = message_queue.get()  # Infinite timeout for now

            if message_type == MessageType.VISION:
                game.add_new_state(message)
            elif message_type == MessageType.REF:
                pass
            packet_builder()
            
    except KeyboardInterrupt:
        print("Stopping main program.")


if __name__ == "__main__":
    logger.level = logging.DEBUG
    main()
