import time
import threading
import numpy as np
from typing import Tuple, Optional, Dict, List, Union

from global_utils.math_utils import rotate_vector
from team_controller.src.data.vision_receiver import VisionDataReceiver
from motion_planning.src.pid.pid import PID
from team_controller.src.config.settings import (
    PID_PARAMS,
    LOCAL_HOST,
    YELLOW_TEAM_SIM_PORT,
    BLUE_TEAM_SIM_PORT,
    YELLOW_START,
)
from team_controller.src.utils import network_manager

from team_controller.src.generated_code.ssl_simulation_robot_control_pb2 import (
    RobotControl,
)
from team_controller.src.generated_code.ssl_simulation_robot_feedback_pb2 import (
    RobotControlResponse, RobotFeedback
)

# TODO: To be moved to a High-level Descision making repo


class SimRobotController:
    def __init__(
        self,
        address=LOCAL_HOST,
        port=(YELLOW_TEAM_SIM_PORT, BLUE_TEAM_SIM_PORT),
        debug=False,
    ):

        self.out_packet = RobotControl()
        
        self.net_yellow = network_manager.NetworkManager(address=(address, port[0]))
        self.net_blue = network_manager.NetworkManager(address=(address, port[1]))
        
        self.yellow_robot_info = RobotControlResponse()
        self.blue_robot_info = RobotControlResponse()

        self.debug = debug

    def send_robot_commands(self, team_is_yellow: bool) -> None:
        """
        Sends the robot commands to the appropriate team (yellow or blue).

        Args:
            team_is_yellow (bool): True if the team is yellow, False if the team is blue.
        """
        if self.debug:
            print(f"Sending Robot Commands")
            
        if team_is_yellow:   
            data = self.net_yellow.send_command(self.out_packet, is_sim=True)
            if data:
                self.yellow_robot_info = RobotControlResponse()
                self.yellow_robot_info.ParseFromString(data)
            self.out_packet.Clear()
        else:
            data = self.net_blue.send_command(self.out_packet, is_sim=True)
            if data:
                self.blue_robot_info = RobotControlResponse()
                self.blue_robot_info.ParseFromString(data)
            self.out_packet.Clear()
        
    def add_robot_command(self, command: Dict) -> None:
        """
        Adds a robot command to the out_packet.

        Args:
            command (dict): A dictionary containing the robot command with keys 'id', 'xvel', 'yvel', and 'wvel'.
        """
        robot = self.out_packet.robot_commands.add()
        robot.id = command["id"]
        local_vel = robot.move_command.local_velocity
        local_vel.forward = command["xvel"]
        local_vel.left = command["yvel"]
        local_vel.angular = command["wvel"]
        # print(f"Robot {command['id']} command: ({command['xvel']:.3f}, {command['yvel']:.3f}, {command['wvel']:.3f})")
    
    def robot_has_ball(self, robot_id: int, team_is_yellow: bool) -> bool:
        """
        Checks if the specified robot has the ball.

        Args:
            robot_id (int): The ID of the robot.
            team_is_yellow (bool): True if the team is yellow, False if the team is blue.

        Returns:
            bool: True if the robot has the ball, False otherwise.
        """
        if team_is_yellow:
            response = self.yellow_robot_info
        else:
            response = self.blue_robot_info
        
        for robot_feedback in response.feedback:
            if robot_feedback.HasField("dribbler_ball_contact") and robot_feedback.id == robot_id:
                if robot_feedback.dribbler_ball_contact:
                    if self.debug:
                        print(f"Robot: {robot_id}: HAS the Ball")
                    return True
                else:
                    return False
