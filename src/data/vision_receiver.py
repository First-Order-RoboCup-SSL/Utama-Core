import threading
import time
from typing import Dict, Optional, Tuple, List

from utils import network_manager
from config.settings import MULTICAST_GROUP, VISION_PORT, NUM_ROBOTS

from generated_code.ssl_vision_wrapper_pb2 import SSL_WrapperPacket


class VisionDataReceiver:
    """
    A class responsible for receiving and managing vision data for robots and the ball in a multi-robot game environment.
    The class interfaces with a network manager to receive packets, which contain positional data for the ball and robots 
    on both teams, and updates the internal data structures accordingly.
    
    Args:
        ip (str): The IP address for receiving multicast vision data. Defaults to MULTICAST_GROUP.
        port (int): The port for receiving vision data. Defaults to VISION_PORT.
    """
    def __init__(self, ip = MULTICAST_GROUP, port = VISION_PORT):
        self.net = network_manager.NetworkManager(address=(ip, port), bind_socket=True)
        
        self.ball_dict: Dict[int, List[float]] = {}
        self.robot_yellow_dict: Dict[int, Optional[List[float]]] = {i: None for i in range(NUM_ROBOTS)}
        self.robot_blue_dict: Dict[int, Optional[List[float]]] = {i: None for i in range(NUM_ROBOTS)}
        
        self.lock = threading.Lock()
    
    def _update_data(self, detection: object) -> None:
        # Update both ball and robot data incrementally.
        self._update_ball_dict(detection)
        self._update_robot_dict(detection)
    
    def _update_ball_dict(self, detection: object) -> None:
        for i, ball in enumerate(detection.balls):
            self.ball_dict[i] = [
                ball.x, 
                ball.y, 
                ball.z if ball.HasField('z') else 0.0   
            ]

    def _update_robot_dict(self, detection: object) -> None:
        # Update both yellow and blue robots from detection.
        self.__update_team_robot_dict(detection.robots_yellow, self.robot_yellow_dict)
        self.__update_team_robot_dict(detection.robots_blue, self.robot_blue_dict)

    def __update_team_robot_dict(self, robots: List[object], robot_dict: Dict[int, Optional[List[float]]]) -> None:
        # Generic method to update robot dictionaries for both teams.
        for robot in robots:
            pos = [robot.x, robot.y]
            if robot.HasField('orientation'):
                pos.append(robot.orientation)
            robot_dict[robot.robot_id] = pos
            
    def _print_frame_info(self, detection: object):
        # TODO: borken t_* values (time not synced with vision)
        
        # t_now = time.time()
        # print(f"Time Now: {t_now:.3f}")
        # print(f"Camera ID={detection.camera_id} FRAME={detection.frame_number} "
        #       f"T_CAPTURE={detection.t_capture:.4f}")
        # print(f"SSL-Vision Processing Latency: "
        #       f"{(detection.t_sent - detection.t_capture) * 1000.0:.3f}ms")
        # print(f"Network Latency: "
        #       f"{(t_now - detection.t_sent) * 1000.0:.3f}ms")
        # print(f"Total Latency: "
        #       f"{(t_now - detection.t_capture) * 1000.0:.3f}ms")
        return 0
    
    def get_robot_dict(self, is_yellow: bool) -> Dict[int, Optional[Tuple[float, float, float]]]:
        """
        Retrieves the current position data for robots on the specified team from the vision data.

        Args:
            is_yellow (bool): If True, retrieves data for the yellow team. If False, retrieves data for the blue team.
        
        Returns:
            Dict[int, Optional[Tuple[float, float, float]]]: A dictionary of all detected robot positions {robot_id: (x, y, orientation)} for the specified team. 
        """
        # Get the dictionary of robots based on the team color.
        with self.lock:
            return self.robot_yellow_dict if is_yellow else self.robot_blue_dict

    def get_ball_dict(self) -> Dict[int,  Optional[Tuple[float, float, float]]]:
        """
        Retrieves the current position data for the ball.

        Returns:
            Dict[int, Optional[Tuple[float, float, float]]]: A dictionary of all detected ball positions (x, y, z).
        """
        # Get the dictionary of ball positions.
        with self.lock:
            return self.ball_dict
        
    def get_game_data(self, debug: bool=False) -> None:
        """
        Continuously receives vision data packets and updates the internal data structures for the game state.
        
        This method runs indefinitely and should typically be started in a separate thread.
        """
        vision_packet = SSL_WrapperPacket()
        while True:
            data = self.net.receive_data()
            if data:
                with self.lock:
                    vision_packet.Clear()  # Clear previous data to avoid memory bloat
                    vision_packet.ParseFromString(data)
                    self._update_data(vision_packet.detection)
            if debug:
                print(f"Robots: {self.get_robot_dict(True)}\n")
            time.sleep(0.0083)  
