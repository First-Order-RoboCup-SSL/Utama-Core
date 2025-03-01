import time
from typing import Deque
from entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from team_controller.src.utils import network_manager
from config.settings import MULTICAST_GROUP, VISION_PORT
from team_controller.src.generated_code.ssl_vision_wrapper_pb2 import SSL_WrapperPacket
import logging


logger = logging.getLogger(__name__)


class VisionReceiver:
    """
    Receives protobuf data from SSL Vision over the network, formats into RawData types and passes it over to the
    VisionProcessor.
    """

    def __init__(self, vision_buffer: Deque[RawVisionData]):
        self.net = network_manager.NetworkManager(
            address=(MULTICAST_GROUP, VISION_PORT), bind_socket=True
        )
        self.vision_buffer = vision_buffer

    def pull_game_data(self) -> None:
        """
        Continuously receives vision data packets and updates the internal data structures for the game state.

        This method runs indefinitely and should be started in a separate thread.
        """
        vision_packet = SSL_WrapperPacket()
        while True:
            data = self.net.receive_data()
            if data is not None:
                vision_packet.Clear()
                vision_packet.ParseFromString(data)

                # Deal with out of order packets by checking timestamp in the buffer
                new_raw_vis_data = self._process_packet(vision_packet.detection)
                if self.vision_buffer:
                    # Only add this if it is more recent
                    if new_raw_vis_data.ts > self.vision_buffer[0].ts:
                        self.vision_buffer.append(new_raw_vis_data)

                # Logging
                self._count_objects_detected(vision_packet.detection)
                self._print_frame_info(vision_packet.detection)

    def _process_packet(
        self, detection_frame: object
    ):  # detection_frame = protobuf packet detection
        return RawVisionData(
            ts=detection_frame.t_capture,
            yellow_robots=[
                RawRobotData(
                    robot.robot_id,
                    robot.x / 1000,
                    robot.y / 1000,
                    robot.orientation,
                    robot.confidence,
                )
                for robot in detection_frame.robots_yellow
            ],
            blue_robots=[
                RawRobotData(
                    robot.robot_id,
                    robot.x / 1000,
                    robot.y / 1000,
                    robot.orientation,
                    robot.confidence,
                )
                for robot in detection_frame.robots_blue
            ],
            balls=[
                RawBallData(
                    ball.x / 1000, ball.y / 1000, ball.z / 1000, ball.confidence
                )
                for ball in detection_frame.balls
            ],
            camera_id=detection_frame.camera_id,
        )

    def _print_frame_info(self, latency: float, detection: object) -> None:
        logger.debug(f"Time Now: {time.time():.3f}s")
        logger.debug(
            f"Camera ID={detection.camera_id} FRAME={detection.frame_number} "
            f"T_CAPTURE={detection.t_capture:.4f}s"
            f" T_SENT={detection.t_sent:.4f}s"
        )
        logger.debug(
            f"SSL-Vision Processing Latency: "
            f"{(detection.t_sent - detection.t_capture) * 1000.0:.3f}ms"
        )
        logger.debug(f"Our Processing Latency: {latency * 1000.0:.3f}ms")

    def _count_objects_detected(self, vision_packet_detect):
        num_yellow_robots = 0
        num_blue_robots = 0
        num_balls = 0
        for _ in range(len(vision_packet_detect.robots_yellow)):
            num_yellow_robots += 1

        for _ in range(len(vision_packet_detect.robots_blue)):
            num_blue_robots += 1

        for _ in range(len(vision_packet_detect.balls)):
            num_balls += 1

        logger.debug(
            f"num of yellow robots detected: {num_yellow_robots}, blue robots detected: {num_blue_robots}, num of balls detected: {num_balls} \n"
        )
