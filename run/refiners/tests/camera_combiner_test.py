from unittest.mock import MagicMock

from entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from entities.game.game_frame import GameFrame
from run.refiners.position import CameraCombiner


def test_combine_same_robots_produces_same():
    mockGame = MagicMock(spec=GameFrame)
    raw_yellow = [RawRobotData(0, -1, -10, 0, 1)]
    raw_blue = [RawRobotData(0, -100, -1000, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 1)

    combined_data = CameraCombiner().combine_cameras(mockGame, [raw_vision_data_cam1, raw_vision_data_cam2])

    for i in range(len(combined_data.yellow_robots)):
        assert combined_data.yellow_robots[i].x == raw_yellow[i].x
        assert combined_data.yellow_robots[i].y == raw_yellow[i].y
        assert combined_data.yellow_robots[i].orientation == raw_yellow[i].orientation

    for i in range(len(combined_data.blue_robots)):
        assert combined_data.blue_robots[i].x == raw_blue[i].x
        assert combined_data.blue_robots[i].y == raw_blue[i].y
        assert combined_data.blue_robots[i].orientation == raw_blue[i].orientation


def test_combine_with_one_camera_empty():
    mockGame = MagicMock(spec=GameFrame)
    raw_yellow = [RawRobotData(0, -1, -10, 0, 1)]
    raw_blue = [RawRobotData(0, -100, -1000, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, [], [], [], 1)

    combined_data = CameraCombiner().combine_cameras(mockGame, [raw_vision_data_cam1, raw_vision_data_cam2])

    for i in range(len(combined_data.yellow_robots)):
        assert combined_data.yellow_robots[i].x == raw_yellow[i].x
        assert combined_data.yellow_robots[i].y == raw_yellow[i].y
        assert combined_data.yellow_robots[i].orientation == raw_yellow[i].orientation

    for i in range(len(combined_data.blue_robots)):
        assert combined_data.blue_robots[i].x == raw_blue[i].x
        assert combined_data.blue_robots[i].y == raw_blue[i].y
        assert combined_data.blue_robots[i].orientation == raw_blue[i].orientation


def test_combine_proximity_multiple_balls():
    c = CameraCombiner()

    balls = {
        0: [RawBallData(0, 0, 0, 1), RawBallData(5, 5, 0, 1)],
        1: [RawBallData(0.02, 0.03, 0, 1), RawBallData(1, 1, 0, 1)],
    }
    balls = c._combine_balls_by_proximity(balls)
    assert len(balls) == 4


def test_combine_filters_low_confidence():
    balls = {
        0: [RawBallData(0, 0, 0, 0.05), RawBallData(5, 5, 0, 0.05)],
        1: [RawBallData(0.02, 0.03, 0, 0.05)],
    }
    assert len(CameraCombiner()._combine_balls_by_proximity(balls)) == 0


def test_combine_filters_removing_one_low_confidence_correctly():
    # Filtered ball should not contribute to result
    original = RawBallData(0, 0, 0, 1)
    balls = {0: [original], 1: [RawBallData(0.02, 0.03, 0, 0.05)]}
    result = CameraCombiner()._combine_balls_by_proximity(balls)
    assert len(result) == 1
    assert result[0] == original


def test_combine_with_both_camera_empty_gives_empty():
    mockGame = MagicMock(spec=GameFrame)
    raw_vision_data_cam2 = RawVisionData(0, [], [], [], 1)
    raw_vision_data_cam1 = RawVisionData(0, [], [], [], 0)
    combined_data = CameraCombiner().combine_cameras(mockGame, [raw_vision_data_cam1, raw_vision_data_cam2])

    assert combined_data.yellow_robots == []
    assert combined_data.blue_robots == []
    assert combined_data.balls == []


if __name__ == "__main__":
    test_combine_same_robots_produces_same()
    test_combine_with_one_camera_empty()
    test_combine_with_both_camera_empty_gives_empty()
    test_combine_proximity_multiple_balls()
    test_combine_filters_low_confidence()
