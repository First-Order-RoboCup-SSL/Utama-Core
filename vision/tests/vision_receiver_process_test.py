from dataclasses import dataclass
from entities.data import vision
from vision.vision_receiver import VisionReceiver, RawVisionData, RawRobotData, RawBallData
from collections import deque

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


def test_process_packet_produces_raw_data():
    # Check the receiver decodes the units correctly
        
    d = deque(maxlen=1)
    # Single camera test 
    v = VisionReceiver([d])
    mock_detection = MockDetection(
        t_capture=1,
        t_sent=1,
        robots_yellow=[MockRobotData(1000, 2000, 3000, 4000, 5000)], #units here are mm from the cameras
        robots_blue=[MockRobotData(100, 200, 300, 400, 500)],
        balls=[MockBallData(10, 20, 30, 40)],
        camera_id=0
    )
    
    raw_data = v._process_packet(mock_detection)

    assert len(raw_data.yellow_robots) == 1
    assert len(raw_data.blue_robots) == 1
    assert len(raw_data.balls) == 1

    assert raw_data.yellow_robots[0] == RawRobotData(1000, 2, 3, 4000, 5000)
    assert raw_data.blue_robots[0] == RawRobotData(100, 0.2, 0.3, 400, 500)
    assert raw_data.balls[0] == RawBallData(0.01, 0.02, 0.03, 40)


def test_vision_buffer_takes_more_recent():
    # Check the receiver decodes the units correctly
        
    d = deque(maxlen=1)
    vision_buffer = [d]
    # Single camera test 
    v = VisionReceiver(vision_buffer)
    mock_packet1 = MockDetection(
        t_capture=1,
        t_sent=1,
        robots_yellow=[MockRobotData(1000, 2000, 3000, 4000, 5000)], #units here are mm from the cameras
        robots_blue=[MockRobotData(100, 200, 300, 400, 500)],
        balls=[MockBallData(10, 20, 30, 40)],
        camera_id=0
    )

    mock_packet2 = MockDetection(
        t_capture=2,
        t_sent=1,
        robots_yellow=[MockRobotData(10000, 20000, 30000, 40000, 50000)], #units here are mm from the cameras
        robots_blue=[],
        balls=[],
        camera_id=0
    )
    v._add_detection_to_buffer(mock_packet1)
    v._add_detection_to_buffer(mock_packet2)
    
    # We should get the more recent packet
    exp = v._process_packet(mock_packet2)
    assert len(vision_buffer[0]) == 1
    assert vision_buffer[0][0] == exp
    assert vision_buffer[0][0].ts == 2


if __name__=="__main__":
    test_process_packet_produces_raw_data()
    test_vision_buffer_takes_more_recent()