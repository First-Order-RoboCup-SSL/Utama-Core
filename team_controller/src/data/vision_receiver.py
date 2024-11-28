import threading
import queue
import time
from team_controller.src.data.message_enum import MessageType
from typing import List, Optional
from entities.data.vision import BallData, RobotData, FrameData, TeamRobotCoords
from team_controller.src.data.base_receiver import BaseReceiver
from team_controller.src.utils import network_manager
from team_controller.src.config.settings import MULTICAST_GROUP, VISION_PORT, NUM_ROBOTS
from team_controller.src.generated_code.ssl_vision_wrapper_pb2 import SSL_WrapperPacket


class VisionDataReceiver(BaseReceiver):
    """
    A class responsible for receiving and managing vision data for robots and the ball in a multi-robot game environment.
    The class interfaces with a network manager to receive packets, which contain positional data for the ball and robots
    on both teams, and passes decoded messages (object positions) over the provided message_queue.

    Args:
        ip (str): The IP address for receiving multicast vision data. Defaults to MULTICAST_GROUP.
        port (int): The port for receiving vision data. Defaults to VISION_PORT.
    """
    def __init__(
        self,
        messsage_queue: queue.SimpleQueue,
        ip=MULTICAST_GROUP,
        port=VISION_PORT,
        n_yellow_robots: int = 6,
        n_blue_robots: int = 6,
        debug=False,
    ):
        super().__init__(messsage_queue) # Setup the message queue

        self.net = network_manager.NetworkManager(address=(ip, port), bind_socket=True)
        self.time_received = None
        self.ball_pos: List[BallData] = None
        self.robots_yellow_pos: List[RobotData] = [None] * n_yellow_robots
        self.robots_blue_pos: List[RobotData] = [None] * n_blue_robots

        self.debug = debug

    def _update_data(self, detection: object) -> None:
        # Update both ball and robot data incrementally.
        self._update_ball_pos(detection)
        self._update_robots_pos(detection)

        # Put the latest game state into the thread-safe queue which will wake up
        # main if it was empty.
        # TODO: we should modify how Game is updated. Instead of appending to the records list, we should really keep any data we don't have updates for.
        self._message_queue.put_nowait((MessageType.VISION, FrameData(
            self.time_received,
            self.robots_yellow_pos,
            self.robots_blue_pos,
            self.ball_pos,
        )))

    def _update_ball_pos(self, detection: object) -> None:

        if not detection.balls:
            return

        ball_pos = []
        for _, ball in enumerate(detection.balls):
            ball_pos.append(
                BallData(ball.x, ball.y, ball.z if ball.HasField("z") else 0.0)
            )
        self.ball_pos = ball_pos

    def _update_robots_pos(self, detection: object) -> None:
        # Update both yellow and blue robots from detection.
        self.__update_team_robots_pos(detection.robots_yellow, self.robots_yellow_pos)
        self.__update_team_robots_pos(detection.robots_blue, self.robots_blue_pos)

    def __update_team_robots_pos(
        self,
        robots_data: object,
        robots: List[RobotData],
    ) -> None:
        # Generic method to update robots for both teams.
        for robot in robots_data:
            if 0 <= robot.robot_id < len(robots):
                robots[robot.robot_id] = RobotData(
                    robot.x,
                    robot.y,
                    robot.orientation if robot.HasField("orientation") else 0,
                )
                # TODO: When do we not have orientation?

    def _print_frame_info(self, t_received: float, detection: object) -> None:
        t_now = time.time()
        print(f"Time Now: {t_now:.3f}s")
        print(
            f"Camera ID={detection.camera_id} FRAME={detection.frame_number} "
            f"T_CAPTURE={detection.t_capture:.4f}s"
            f" T_SENT={detection.t_sent:.4f}s"
        )
        print(
            f"SSL-Vision Processing Latency: "
            f"{(detection.t_sent - detection.t_capture) * 1000.0:.3f}ms"
        )
        print(
            f"Total Latency: "
            f"{((t_now - t_received) + (detection.t_sent - detection.t_capture)) * 1000.0:.3f}ms"
        )

    def pull_game_data(self) -> None:
        """
        Continuously receives vision data packets and updates the internal data structures for the game state.

        This method runs indefinitely and should typically be started in a separate thread.
        """
        vision_packet = SSL_WrapperPacket()
        while True:
            t_received = time.time()
            self.time_received = t_received
            data = self.net.receive_data()
            if data is not None:
                vision_packet.Clear()  # Clear previous data to avoid memory bloat
                vision_packet.ParseFromString(data)
                self._update_data(vision_packet.detection)
            if self.debug:
                self._print_frame_info(t_received, vision_packet.detection)
            time.sleep(0.0083) # TODO : Block on data?

    # MOVE INTO GAME  
  
    # # Implemented already  
    # def get_robot_by_id(self, is_yellow: bool, robot_id: int) -> RobotData:
    #         """
    #         Retrieves the position data for a specific robot by ID.
    #         Args:
    #             is_yellow (bool): If True, retrieves data for the yellow team; otherwise, for the blue team.
    #             robot_id (int): The ID of the robot.
    #         Returns:
    #             RobotData: The position data of the specified robot.
    #         """
    #         with self.lock:
    #             robots = self.robots_yellow_pos if is_yellow else self.robots_blue_pos
    #             if 0 <= robot_id < len(robots) and robots[robot_id] is not None:
    #                 return robots[robot_id]
    #             else:
    #                 return None  # TODO: Or raise an exception.    
    
    #            
    # def get_closest_robot_at_point(self, is_yellow: bool, x: float, y: float) -> RobotData:
    #     """
    #     Finds the robot closest to a given point.
        
    #     Args:
    #         is_yellow (bool): If True, searches within the yellow team; otherwise, within the blue team.
    #         x (float): The x-coordinate of the point.
    #         y (float): The y-coordinate of the point.

    #     Returns:
    #         RobotData: The position data of the closest robot.
    #     """
    #     with self.lock:
    #         robots = self.robots_yellow_pos if is_yellow else self.robots_blue_pos
    #         min_distance = float('inf')
    #         closest_robot = None
    #         for robot in robots:
    #             if robot is not None:
    #                 distance = ((robot.x - x) ** 2 + (robot.y - y) ** 2) ** 0.5
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     closest_robot = robot
    #     # TODO: Haven't been tested 
    #     return closest_robot   
    
    # # Implemented Already
    # def get_ball_velocity(self) -> Optional[tuple]: # UNUSED
    #     """
    #     Calculates the ball's velocity based on position changes over time.

    #     Returns:
    #         tuple: The velocity components (vx, vy).
    #     """
    #     # TODO Find a method to store the data and get velocity. --> self.previour_ball_pos
    #     with self.lock:
    #         if len(self.history) < 2:
    #             # Not suffucient data to extrapolate velocity
    #             return None
    #         # Otherwise get the previous and current frames
    #         previous_frame = self.history[-2]
    #         current_frame = self.history[-1]
            
    #         previous_ball_pos = previous_frame.ball[0] #TODO don't always take first ball pos
    #         ball_pos = current_frame.ball[0]
    #         previous_time_received = previous_frame.ts
    #         time_received = current_frame.ts

    #         # Latest frame should always be ahead of last one    
    #         if time_received < previous_time_received:
    #             # TODO log a warning
    #             print("Timestamps out of order for vision data ")
    #             return None        
            
    #         dt = time_received - previous_time_received
    #         vx = (ball_pos.x - previous_ball_pos.x) / dt
    #         vy = (ball_pos.y - previous_ball_pos.y) / dt
    #         return (vx, vy)
