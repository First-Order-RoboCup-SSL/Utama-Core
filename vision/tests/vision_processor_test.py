from entities.data.raw_vision import RawBallData, RawVisionData, RawRobotData
from vision.vision_processor import VisionProcessor
from unittest.mock import Mock

def test_not_ready_until_all_expected_received():
    mock = Mock()
    
    EXPECTED_YELLOW = 4
    EXPECTED_BLUE = 3

    robots = [RawRobotData(id=i, x=2, y=2, orientation=0, confidence=1) for i in range(max(EXPECTED_YELLOW, EXPECTED_BLUE))]

    vision_processor = VisionProcessor(EXPECTED_YELLOW, EXPECTED_BLUE, 1, mock)
    vision_processor.add_new_frame(RawVisionData(
        ts=1,
        yellow_robots=[robots[0]],
        blue_robots=[robots[1]],
        balls=[],
        camera_id=0
    ))
    assert not vision_processor.is_ready()
    vision_processor.add_new_frame(RawVisionData(
        ts=2,
        yellow_robots=robots[:EXPECTED_YELLOW],
        blue_robots=robots[:EXPECTED_BLUE],
        balls=[],
        camera_id=0
    ))
    assert not vision_processor.is_ready()
    vision_processor.add_new_frame(RawVisionData(
        ts=2,
        yellow_robots=robots[:EXPECTED_YELLOW],
        blue_robots=robots[:EXPECTED_BLUE],
        balls=[RawBallData(5, 5, 5, 1)],
        camera_id=0
    ))
    assert vision_processor.is_ready()

# def test_all_expected_can_be_fragmented():


# def test_averages_data_from_multiple_cameras():

# def test_extrapolates_missing_data_after_started():
#     pass