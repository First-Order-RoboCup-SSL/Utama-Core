from collections import deque
from dataclasses import dataclass, replace

from utama_core.data_processing.receivers.vision_receiver import (
    RawBallData,
    RawRobotData,
    VisionReceiver,
)


@dataclass
class MockDetection:
    t_capture: float
    t_sent: float
    robots_yellow: list[RawRobotData]
    robots_blue: list[RawRobotData]
    balls: list[RawBallData]
    camera_id: int


@dataclass
class MockRobotData:
    robot_id: int
    x: float
    y: float
    orientation: float
    confidence: float


@dataclass
class MockBallData:
    x: float
    y: float
    z: float
    confidence: float


MOCK_DETECTION1 = MockDetection(
    t_capture=1,
    t_sent=1,
    robots_yellow=[MockRobotData(1000, 2000, 3000, 4000, 5000)],  # units here are mm from the cameras
    robots_blue=[MockRobotData(100, 200, 300, 400, 500)],
    balls=[MockBallData(10, 20, 30, 40)],
    camera_id=0,
)
MOCK_DETECTION2 = MockDetection(
    t_capture=2,
    t_sent=1,
    robots_yellow=[MockRobotData(10000, 20000, 30000, 40000, 50000)],  # units here are mm from the cameras
    robots_blue=[],
    balls=[],
    camera_id=0,
)


def test_process_packet_produces_raw_data():
    # Check the receiver decodes the units correctly

    d = deque(maxlen=1)
    # Single camera test
    v = VisionReceiver([d])

    raw_data = v._process_packet(MOCK_DETECTION1)

    assert len(raw_data.yellow_robots) == 1
    assert len(raw_data.blue_robots) == 1
    assert len(raw_data.balls) == 1

    assert raw_data.yellow_robots[0] == RawRobotData(1000, 2, 3, 4000, 5000)
    assert raw_data.blue_robots[0] == RawRobotData(100, 0.2, 0.3, 400, 500)
    assert raw_data.balls[0] == RawBallData(0.01, 0.02, 0.03, 40)


def test_single_camera_takes_more_recent_frame():
    # Single camera test
    # Check the receiver decodes the units correctly

    d = deque(maxlen=1)
    vision_buffer = [d]
    v = VisionReceiver(vision_buffer)
    v._add_detection_to_buffer(MOCK_DETECTION1)
    v._add_detection_to_buffer(MOCK_DETECTION2)

    # We should get the more recent packet
    exp = v._process_packet(MOCK_DETECTION2)
    assert len(vision_buffer[0]) == 1
    assert vision_buffer[0][0] == exp
    assert vision_buffer[0][0].ts == 2


def test_multiple_cameras_put_frames_to_buffer():
    # Check the receiver decodes the units correctly
    vision_buffer = [deque(maxlen=1) for _ in range(4)]
    # Single camera test
    v = VisionReceiver(vision_buffer)
    mock_detection3 = replace(MOCK_DETECTION1, camera_id=1)
    mock_detection4 = replace(MOCK_DETECTION1, camera_id=2)
    mock_detection5 = replace(MOCK_DETECTION1, camera_id=3)

    camera_detections = [
        MOCK_DETECTION1,
        mock_detection3,
        mock_detection4,
        mock_detection5,
    ]
    for d in camera_detections:
        v._add_detection_to_buffer(d)

    for buffer_id in range(4):
        assert len(vision_buffer[buffer_id]) == 1
        assert vision_buffer[buffer_id][0].camera_id == buffer_id
        assert vision_buffer[buffer_id][0] == v._process_packet(camera_detections[buffer_id])


if __name__ == "__main__":
    test_process_packet_produces_raw_data()
    test_single_camera_takes_more_recent_frame()
    test_multiple_cameras_put_frames_to_buffer()
