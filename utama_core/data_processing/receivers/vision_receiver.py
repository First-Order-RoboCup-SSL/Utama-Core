import logging
import threading
from typing import Callable, Deque, Dict, List, Optional, Tuple

from utama_core.config.settings import MULTICAST_GROUP, VISION_PORT
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)
from utama_core.team_controller.src.utils import network_manager

logger = logging.getLogger(__name__)


class VisionReceiver:
    """Receives protobuf data from SSL Vision over the network, formats into RawData types and passes it over to the
    VisionProcessor."""

    def __init__(
        self,
        vision_buffers: List[Deque[RawVisionData]],
        on_geometry: Optional[Callable] = None,
        stop_event: Optional[threading.Event] = None,
        on_packet: Optional[Callable[[bytes], None]] = None,
    ):
        self.net = network_manager.NetworkManager(address=(MULTICAST_GROUP, VISION_PORT), bind_socket=True)
        self.vision_buffers = vision_buffers
        self._on_geometry = on_geometry
        self._geometry_fired = False
        self._stop_event = stop_event
        self._on_packet = on_packet  # every packet as received, before parsing (`GameLogWriter`)
        # camera id -> (x, y, height) in m, from every geometry packet's calibrations
        self._camera_positions: Dict[int, Tuple[float, float, float]] = {}
        # Parked exception from the geometry callback — re-raised in the main
        # thread by StrategyRunner._run_step() so the process exits cleanly.
        self.thread_exception: Optional[BaseException] = None

    def _add_detection_to_buffer(self, detection_frame: object) -> None:
        # Deal with out of order packets by checking timestamp in the buffer
        new_raw_vis_data = self._process_packet(detection_frame)
        if self.vision_buffers[new_raw_vis_data.camera_id]:
            # Only add this if it is more recent
            if new_raw_vis_data.ts > self.vision_buffers[new_raw_vis_data.camera_id][0].ts:
                self.vision_buffers[new_raw_vis_data.camera_id].append(new_raw_vis_data)
        else:
            self.vision_buffers[new_raw_vis_data.camera_id].append(new_raw_vis_data)

    def pull_game_data(self) -> None:
        """Continuously receives vision data packets and updates the internal data structures for the game state.

        This method runs indefinitely and should be started in a separate thread.
        """
        vision_packet = SSL_WrapperPacket()
        while True:
            data = self.net.receive_data()
            if data is not None and not self._handle_packet(vision_packet, data):
                return

    def _handle_packet(self, vision_packet: SSL_WrapperPacket, data: bytes) -> bool:
        """Parse one wrapper packet into `vision_packet` and buffer its detection. Returns False
        when the geometry callback failed and the receiver should stop."""
        if self._on_packet:
            self._on_packet(data)
        vision_packet.Clear()
        vision_packet.ParseFromString(data)
        if self._on_geometry and not self._geometry_fired and vision_packet.HasField("geometry"):
            try:
                self._on_geometry(vision_packet.geometry.field)
            except Exception as exc:
                self.thread_exception = exc
                if self._stop_event is not None:
                    self._stop_event.set()
                return False
            self._geometry_fired = True
        if vision_packet.HasField("geometry"):
            for calib in vision_packet.geometry.calib:
                # A height of 0 is a camera not calibrated yet: its view-angle weights would all be 0.
                if calib.HasField("derived_camera_world_tz") and calib.derived_camera_world_tz > 0:
                    self._camera_positions[calib.camera_id] = (
                        calib.derived_camera_world_tx / 1000,
                        calib.derived_camera_world_ty / 1000,
                        calib.derived_camera_world_tz / 1000,
                    )
        # SSL-Vision and grSim send geometry in packets of its own: an absent detection reads as an
        # empty camera-0 frame captured at t=0, which would drag the combined frame's timestamp down.
        if vision_packet.HasField("detection"):
            self._add_detection_to_buffer(vision_packet.detection)
        return True

    def _process_packet(self, detection_frame: object):  # detection_frame = protobuf packet detection
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
                RawBallData(ball.x / 1000, ball.y / 1000, ball.z / 1000, ball.confidence)
                for ball in detection_frame.balls
            ],
            camera_id=detection_frame.camera_id,
            camera_position=self._camera_positions.get(detection_frame.camera_id),
        )
