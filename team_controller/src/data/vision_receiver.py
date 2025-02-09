import threading
import queue
import time
from team_controller.src.data.message_enum import MessageType
from typing import List, Optional, Tuple
from entities.data.vision import BallData, RobotData, FrameData, TeamRobotCoords
from team_controller.src.data.base_receiver import BaseReceiver
from team_controller.src.utils import network_manager
from team_controller.src.config.settings import MULTICAST_GROUP, VISION_PORT
from team_controller.src.generated_code.ssl_vision_wrapper_pb2 import SSL_WrapperPacket
import logging

logger = logging.getLogger(__name__)


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
        n_cameras=4,
    ):
        super().__init__(messsage_queue)  # Setup the message queue

        self.net = network_manager.NetworkManager(address=(ip, port), bind_socket=True)
        self.time_received = None
        self.ball_pos: List[BallData] = None
        self.robots_yellow_pos: List[RobotData] = [None] * n_yellow_robots
        self.robots_blue_pos: List[RobotData] = [None] * n_blue_robots
        self.camera_frames = [None for i in range(n_cameras)]  # TODO: Use GEOMETRY
        self.frames_recvd = 0
        self.n_cameras = n_cameras

    def _update_data(self, detection: object) -> None:  # SSL_DetectionPacket
        # Update both ball and robot data incrementally.

        # TODO: flush robots_yellow_pos, flush robots_blue_pos with None before updating?

        self._update_ball_pos(detection)
        self._update_robots_pos(detection)

        self.frames_recvd += 1

        new_frame = FrameData(
            self.time_received,
            self.robots_yellow_pos,
            self.robots_blue_pos,
            self.ball_pos,
        )

        if all(map(lambda x: x is None, self.robots_yellow_pos)):
            # print("No yellow FOUND")
            pass

        self.camera_frames[detection.camera_id] = new_frame
        if (
            self.frames_recvd % self.n_cameras == 0 and not None in self.camera_frames
        ):  # TODO : Do something more advanced than an average because cameras might not be round robin
            # Put the latest game state into the thread-safe queue which will wake up
            # main if it was empty.
            # TODO: we should modify how Game is updated. Instead of appending to the records list, we should really keep any data we don't have updates for.
            self._message_queue.put_nowait(
                (MessageType.VISION, self._avg_frames(self.camera_frames))
            )

    def _avg_robots(self, rs: List[RobotData]) -> Optional[RobotData]:
        if not rs:
            return None

        tx, ty, to = 0, 0, 0
        for r in rs:
            tx += r.x
            ty += r.y
            to += r.orientation

        return RobotData(tx / len(rs), ty / len(rs), to / len(rs))

    def _avg_balls(self, bs: List[BallData]) -> Optional[BallData]:
        if not bs:
            return None

        tx, ty, tz = 0, 0, 0
        for r in bs:
            tx += r.x
            ty += r.y
            tz += r.z

        return BallData(
            tx / len(bs),
            ty / len(bs),
            tz / len(bs),
            min(map(lambda x: x.confidence, bs)),
        )

    def _avg_frames(self, frames: List[FrameData]) -> FrameData:
        frames = [*filter(lambda x: x.ball is not None, frames)]
        ts = 0
        yellow_captured = [[] for _ in range(11)]
        blue_captured = [[] for _ in range(11)]
        ball_captured = [[] for _ in range(11)]

        for frame in frames:
            for ind, yr in enumerate(frame.yellow_robots):
                if yr is not None:
                    yellow_captured[ind].append(yr)
            for ind, br in enumerate(frame.blue_robots):
                if br is not None:
                    blue_captured[ind].append(br)
            for ind, b in enumerate(frame.ball):
                if b is not None:
                    ball_captured[ind].append(b)
            ts += frame.ts

        avg_yellows = list(map(self._avg_robots, yellow_captured))
        avg_blues = list(map(self._avg_robots, blue_captured))
        avg_balls = list(map(self._avg_balls, ball_captured))

        # Trims number of robots in frame to number we expect (num_friendly, num_enemy) currently done to 6
        return FrameData(ts, avg_yellows[:-5], avg_blues[:-5], avg_balls[:-10])

    def _update_ball_pos(self, detection: object) -> None:

        if not detection.balls:
            return

        ball_pos = []
        for _, ball in enumerate(detection.balls):
            ball_pos.append(
                BallData(
                    ball.x / 1000,
                    ball.y / 1000,
                    (ball.z / 1000) if ball.HasField("z") else 0.0,
                    ball.confidence,
                )
            )
        # sorted by ball confidence
        sorted_balls = sorted(ball_pos, key=lambda ball: ball.confidence)
        self.ball_pos = sorted_balls

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
                    robot.x / 1000,
                    robot.y / 1000,
                    robot.orientation if robot.HasField("orientation") else 0,
                )
                # TODO: When do we not have orientation?

    def _print_frame_info(self, t_received: float, detection: object) -> None:
        t_now = time.time()
        logger.debug(f"Time Now: {t_now:.3f}s")
        logger.debug(
            f"Camera ID={detection.camera_id} FRAME={detection.frame_number} "
            f"T_CAPTURE={detection.t_capture:.4f}s"
            f" T_SENT={detection.t_sent:.4f}s"
        )
        logger.debug(
            f"SSL-Vision Processing Latency: "
            f"{(detection.t_sent - detection.t_capture) * 1000.0:.3f}ms"
        )
        logger.debug(
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
            data = self.net.receive_data()
            t_received = time.time()  # TODO: DUBIOUS because of thread scheduling?
            self.time_received = t_received
            if data is not None:
                vision_packet.Clear()  # Clear previous data to avoid memory bloat
                vision_packet.ParseFromString(data)
                self.print_detection_info(vision_packet.detection)
                self._update_data(vision_packet.detection)
                # print(vision_packet.detection)

            # self._print_frame_info(t_received, vision_packet.detection)
            # time.sleep(0.0083) # TODO : Block on data?

    def print_detection_info(self, vision_packet_detect):
        num_yellow_robots = 0
        num_blue_robots = 0
        num_balls = 0
        for _ in range(len(vision_packet_detect.robots_yellow)):
            num_yellow_robots += 1

        for _ in range(len(vision_packet_detect.robots_blue)):
            num_blue_robots += 1

        for _ in range(len(vision_packet_detect.balls)):
            num_balls += 1

        # print(
        #     f"num of yellow robots detected: {num_yellow_robots}, blue robots detected: {num_blue_robots}"
        # )
        # print(f"num of balls detected: {num_balls} \n")
