import time
from typing import Deque, List
from entities.data import vision
from entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from team_controller.src.utils import network_manager
from config.settings import MULTICAST_GROUP, VISION_PORT
from team_controller.src.generated_code.ssl_vision_wrapper_pb2 import SSL_WrapperPacket
import logging

from collections import deque

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class VisionReceiver:
    """
    Receives protobuf data from SSL Vision over the network, formats into RawData types and passes it over to the
    VisionProcessor.
    """

    def __init__(self, vision_buffers: List[Deque[RawVisionData]]):
        self.net = network_manager.NetworkManager(
            address=(MULTICAST_GROUP, VISION_PORT), bind_socket=True
        )
        self.vision_buffers = vision_buffers
        self.packet_timestamps = deque()
        self.fps_print_interval = 1  # seconds
        self.last_fps_print_time = time.time()
        self.prev_frame_num = 0


    def _add_detection_to_buffer(self, detection_frame: object) -> None:
        # Deal with out of order packets by checking timestamp in the buffer
        new_raw_vis_data = self._process_packet(detection_frame)
        if self.vision_buffers[new_raw_vis_data.camera_id]:
            # Only add this if it is more recent
            if new_raw_vis_data.ts > self.vision_buffers[new_raw_vis_data.camera_id][0].ts:
                self.vision_buffers[new_raw_vis_data.camera_id].append(new_raw_vis_data)
        else:
            self.vision_buffers[new_raw_vis_data.camera_id].append(new_raw_vis_data)

    def pull_game_data(self, fps = True) -> None:
        """
        Continuously receives vision data packets and updates the internal data structures for the game state.

        This method runs indefinitely and should be started in a separate thread.
        """
        vision_packet = SSL_WrapperPacket()
        while True:
            recv_time = time.time()
            data = self.net.receive_data()
            if data is not None:
                vision_packet.Clear()
                vision_packet.ParseFromString(data)
                # print(vision_packet.detection)
                self.prev_frame_num = vision_packet.detection.frame_number
                self._add_detection_to_buffer(vision_packet.detection)
                proc_latency = time.time()-recv_time
                # Logging
                # self._count_objects_detected(vision_packet.detection)
                # self._print_frame_info(proc_latency, vision_packet.detection)
                
                if fps:
                    # --- FPS Tracking ---
                    self.packet_timestamps.append(recv_time)
                    # Remove timestamps older than 1 second
                    while self.packet_timestamps and self.packet_timestamps[0] < recv_time - 1.0:
                        self.packet_timestamps.popleft()

                    if recv_time - self.last_fps_print_time >= self.fps_print_interval:
                        fps = len(self.packet_timestamps)
                        cameras = 4
                        print(f"Current Vision FPS: {fps/cameras}")
                        self.last_fps_print_time = recv_time
    
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
