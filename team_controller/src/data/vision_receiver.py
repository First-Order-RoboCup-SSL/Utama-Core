import threading
import time
from typing import List, NamedTuple
from collections import namedtuple

from team_controller.src.utils import network_manager
from team_controller.src.config.settings import MULTICAST_GROUP, VISION_PORT, NUM_ROBOTS

from team_controller.src.generated_code.ssl_vision_wrapper_pb2 import SSL_WrapperPacket

Ball = namedtuple("Ball", ["x", "y", "z"])
Robot = namedtuple("Robot", ["x", "y", "orientation"])


class VisionDataReceiver:
    """
    A class responsible for receiving and managing vision data for robots and the ball in a multi-robot game environment.
    The class interfaces with a network manager to receive packets, which contain positional data for the ball and robots
    on both teams, and updates the internal data structures accordingly.

    Args:
        ip (str): The IP address for receiving multicast vision data. Defaults to MULTICAST_GROUP.
        port (int): The port for receiving vision data. Defaults to VISION_PORT.
    """

    def __init__(
        self,
        ip=MULTICAST_GROUP,
        port=VISION_PORT,
        n_yellow_robots: int = 6,
        n_blue_robots: int = 6,
        debug=False,
    ):
        self.net = network_manager.NetworkManager(address=(ip, port), bind_socket=True)

        self.ball_pos: List[Ball] = None
        self.robots_yellow_pos: List[Robot] = [None] * n_yellow_robots
        self.robots_blue_pos: List[Robot] = [None] * n_blue_robots

        self.lock = threading.Lock()
        self.debug = debug

    def _update_data(self, detection: object) -> None:
        # Update both ball and robot data incrementally.
        self._update_ball_pos(detection)
        self._update_robots_pos(detection)

    def _update_ball_pos(self, detection: object) -> None:

        if not detection.balls:
            return

        ball_pos = []
        for i, ball in enumerate(detection.balls):
            ball_pos.append(Ball(ball.x, ball.y, ball.z if ball.HasField("z") else 0.0))
        self.ball_pos = ball_pos

    def _update_robots_pos(self, detection: object) -> None:
        # Update both yellow and blue robots from detection.
        self.__update_team_robots_pos(detection.robots_yellow, self.robots_yellow_pos)
        self.__update_team_robots_pos(detection.robots_blue, self.robots_blue_pos)

    def __update_team_robots_pos(
        self,
        robots_data: object,
        robots: List[NamedTuple],
    ) -> None:
        # Generic method to update robots for both teams.
        for robot in robots_data:
            robots[robot.robot_id] = Robot(
                robot.x,
                robot.y,
                robot.orientation if robot.HasField("orientation") else 0,
            )
            # TODO: When do we not have orientation?

    def _print_frame_info(self, detection: object):
        # TODO: broken t_* values (time not synced with vision)

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

    def get_robots_pos(self, is_yellow: bool) -> List[Robot]:
        """
        Retrieves the current position data for robots on the specified team from the vision data.

        Args:
            is_yellow (bool): If True, retrieves data for the yellow team. If False, retrieves data for the blue team.

        Returns:
            List[Robot]: A list of all detected robot positions {robot_id: (x, y, orientation)} for the specified team.
        """
        with self.lock:
            return self.robots_yellow_pos if is_yellow else self.robots_blue_pos

    def get_ball_pos(self) -> Ball:
        """
        Retrieves the current position data for the ball.

        Returns:
            Ball: A named tuple of all detected ball positions (x, y, z).
        """
        with self.lock:
            return self.ball_pos

    def get_game_data(self) -> None:
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
            if self.debug:
                print(f"Robots: {self.get_robots_pos(True)}\n")
                print(f"Ball: {self.get_ball_pos()}\n")
            time.sleep(0.0083)
