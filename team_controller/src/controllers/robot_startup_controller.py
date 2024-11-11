import time
import threading
import numpy as np
from typing import Tuple, Optional, Dict, List, Union

from global_utils.math_utils import rotate_vector
from data.vision_receiver import VisionDataReceiver
from pid.pid import PID
from config.settings import PID_PARAMS, LOCAL_HOST, YELLOW_TEAM_SIM_PORT, BLUE_TEAM_SIM_PORT, YELLOW_START
from utils import network_manager

from generated_code.ssl_simulation_robot_control_pb2 import RobotControl

# TODO: To be moved to a High-level Descision making repo

class StartUpController:
    def __init__(self, vision_receiver: VisionDataReceiver, address=LOCAL_HOST, port=(YELLOW_TEAM_SIM_PORT, BLUE_TEAM_SIM_PORT), debug=False):
        self.vision_receiver = vision_receiver
        
        self.net = network_manager.NetworkManager(address=(address, port[0]))
        
        # TODO: Tune PID parameters further when going from sim to real(it works for Grsim)
        # potentially have a set tunig parameters for each robot
        self.pid_oren = PID(0.0167, 8, -8, 5, 0.01, 0, num_robots=6)
        self.pid_trans = PID(0.0167, 1.5, -1.5, 5, 0.01, 0, num_robots=6)

        self.lock = threading.Lock() 
        
        self.debug = debug 

    def startup(self):
        while True:
            start_time = time.time()
            
            robots, balls = self._get_positions()
                
            if robots and balls:
                out_packet = RobotControl()
                for robot_id, robot_data in robots.items():
                    if robot_data is None:
                        continue
                    target_coords = YELLOW_START[robot_id]
                    command = self._calculate_robot_velocities(robot_id, target_coords, robots, balls, face_ball=True)
                    self._add_robot_command(out_packet, command)
                
                if self.debug:
                    print(out_packet)   
                self.net.send_command(out_packet)
                
            time_to_sleep = max(0, 0.0167 - (time.time() - start_time))
            time.sleep(time_to_sleep)

    def _get_positions(self) -> tuple:
        # Fetch the latest positions of robots and balls with thread locking.
        with self.lock:
            robots = self.vision_receiver.get_robot_dict(is_yellow=True)
            balls = self.vision_receiver.get_ball_dict()
        return robots, balls
    
    def _calculate_robot_velocities(
            self, robot_id: int, target_coords: Union[Tuple[float, float], Tuple[float, float, float]],
            robots: Dict[int, Optional[Tuple[float, float, float]]], balls: Dict[int, Tuple[float, float, float]], face_ball=False
        ) -> Dict[str, float]:
        """
        Calculates the linear and angular velocities required for a robot to move towards a specified target position
        and orientation.

        Args:
            robot_id (int): Unique identifier for the robot.
            target_coords (Tuple[float, float] | Tuple[float, float, float]): Target coordinates the robot should move towards.
                Can be a (x, y) or (x, y, orientation) tuple. If `face_ball` is True, the robot will face the ball instead of 
                using the orientation value in target_coords.
robots (Dict[int, Optional[Tuple[float, float, float]]]): All the Current coordinates of the robots sepateated 
                by thier robot_id which containts a tuple (x, y, orientation).
            balls (Dict[int, Tuple[float, float, float]]): All the Coordinates of the detected balls (int) , typically (x, y, z/height in 3D space).            face_ball (bool, optional): If True, the robot will orient itself to face the ball's position. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary containing the following velocity components:
                - "id" (int): Robot identifier.
                - "xvel" (float): X-axis velocity to move towards the target.
                - "yvel" (float): Y-axis velocity to move towards the target.
                - "wvel" (float): Angular velocity to adjust the robot's orientation.

        The method uses PID controllers to calculate velocities for linear and angular movement. If `face_ball` is set,
        the robot will calculate the angular velocity to face the ball. The resulting x and y velocities are rotated to align
        with the robot's current orientation.
        """
        
        out = {"id": robot_id, "xvel": 0, "yvel": 0, "wvel": 0}

        # Get current positions
        if balls[0] and robots[robot_id]:
            ball_x, ball_y, ball_z = balls[0]
            current_x, current_y, current_oren = robots[robot_id]
        
        target_x, target_y = target_coords[:2]
        
        if face_ball:
            target_oren = np.atan2(ball_y - current_y, ball_x - current_x)
        elif not face_ball and len(target_coords) == 3:
            target_oren = target_coords[2]
            
        # print(f"\nRobot {robot_id} current position: ({current_x:.3f}, {current_y:.3f}, {current_oren:.3f})")
        # print(f"Robot {robot_id} target position: ({target_x:.3f}, {target_y:.3f}, {target_oren:.3f})")
            
        if target_oren != None:
            out["wvel"] = self.pid_oren.calculate(target_oren, current_oren, robot_id, oren=True)

        if target_x != None and target_y != None:
            out["yvel"] = self.pid_trans.calculate(target_y, current_y, robot_id, normalize_range=3000)
            out["xvel"] = self.pid_trans.calculate(target_x, current_x, robot_id, normalize_range=4500)

            out["xvel"], out["yvel"] = rotate_vector(out["xvel"], out["yvel"], current_oren)

        return out
    
    def _add_robot_command(self, out_packet, command) -> None:
        robot = out_packet.robot_commands.add()
        robot.id = command["id"]
        local_vel = robot.move_command.local_velocity
        local_vel.forward = command["xvel"]
        local_vel.left = command["yvel"]
        local_vel.angular = command["wvel"]   
        # print(f"Robot {command['id']} command: ({command['xvel']:.3f}, {command['yvel']:.3f}, {command['wvel']:.3f})")
    