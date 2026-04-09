import math

from utama_core.data_processing.refiners.position import CameraCombiner, VisionBounds
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData

infinite_bounds = VisionBounds(x_min=-math.inf, x_max=math.inf, y_min=-math.inf, y_max=math.inf)


def test_combine_same_robots_produces_same():
    raw_yellow = [RawRobotData(0, -1, -10, 0, 1)]
    raw_blue = [RawRobotData(0, -100, -1000, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 1)

    combined_data = CameraCombiner().combine_cameras(
        [raw_vision_data_cam1, raw_vision_data_cam2],
        infinite_bounds,
    )

    for i in range(len(combined_data.yellow_robots)):
        assert combined_data.yellow_robots[i].x == raw_yellow[i].x
        assert combined_data.yellow_robots[i].y == raw_yellow[i].y
        assert combined_data.yellow_robots[i].orientation == raw_yellow[i].orientation

    for i in range(len(combined_data.blue_robots)):
        assert combined_data.blue_robots[i].x == raw_blue[i].x
        assert combined_data.blue_robots[i].y == raw_blue[i].y
        assert combined_data.blue_robots[i].orientation == raw_blue[i].orientation


def test_combine_with_one_camera_empty():
    raw_yellow = [RawRobotData(0, -1, -10, 0, 1)]
    raw_blue = [RawRobotData(0, -100, -1000, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, [], [], [], 1)

    combined_data = CameraCombiner().combine_cameras([raw_vision_data_cam1, raw_vision_data_cam2], infinite_bounds)

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


def test_combine_with_both_camera_empty_gives_empty():
    raw_vision_data_cam2 = RawVisionData(0, [], [], [], 1)
    raw_vision_data_cam1 = RawVisionData(0, [], [], [], 0)
    combined_data = CameraCombiner().combine_cameras([raw_vision_data_cam1, raw_vision_data_cam2], infinite_bounds)

    assert combined_data.yellow_robots == []
    assert combined_data.blue_robots == []
    assert combined_data.balls == []


def test_combine_filters_low_confidence():
    balls = {
        0: [RawBallData(0, 0, 0, 0.05), RawBallData(5, 5, 0, 0.05)],
        1: [RawBallData(0.02, 0.02, 0, 0.05)],
    }
    assert len(CameraCombiner()._combine_balls_by_proximity(balls)) == 2


# --- Bounds filtering tests ---

tight_bounds = VisionBounds(x_min=-5.0, x_max=5.0, y_min=-3.5, y_max=3.5)


def test_out_of_bounds_yellow_robot_is_filtered():
    in_bounds = RawRobotData(0, 1.0, 1.0, 0, 1)
    out_of_bounds = RawRobotData(1, 10.0, 0.0, 0, 1)  # x > x_max
    frame = RawVisionData(0, [in_bounds, out_of_bounds], [], [], 0)

    result = CameraCombiner().combine_cameras([frame], tight_bounds)

    assert len(result.yellow_robots) == 1
    assert result.yellow_robots[0].id == 0


def test_out_of_bounds_blue_robot_is_filtered():
    in_bounds = RawRobotData(0, -1.0, -1.0, 0, 1)
    out_of_bounds = RawRobotData(1, 0.0, 100.0, 0, 1)  # y > y_max
    frame = RawVisionData(0, [], [in_bounds, out_of_bounds], [], 0)

    result = CameraCombiner().combine_cameras([frame], tight_bounds)

    assert len(result.blue_robots) == 1
    assert result.blue_robots[0].id == 0


def test_out_of_bounds_ball_is_filtered():
    in_bounds_ball = RawBallData(0.0, 0.0, 0, 1)
    out_of_bounds_ball = RawBallData(6.0, 0.0, 0, 1)  # x > x_max
    frame = RawVisionData(0, [], [], [in_bounds_ball, out_of_bounds_ball], 0)

    result = CameraCombiner().combine_cameras([frame], tight_bounds)

    assert len(result.balls) == 1
    assert result.balls[0].x == 0.0


def test_all_robots_out_of_bounds_gives_empty():
    robots = [RawRobotData(i, 50.0 + i, 50.0, 0, 1) for i in range(3)]
    frame = RawVisionData(0, robots, robots, [], 0)

    result = CameraCombiner().combine_cameras([frame], tight_bounds)

    assert result.yellow_robots == []
    assert result.blue_robots == []


def test_robot_exactly_on_boundary_is_included():
    # Boundary is inclusive
    on_boundary = RawRobotData(0, 5.0, 3.5, 0, 1)  # exactly at x_max, y_max
    frame = RawVisionData(0, [on_boundary], [], [], 0)

    result = CameraCombiner().combine_cameras([frame], tight_bounds)

    assert len(result.yellow_robots) == 1
    assert result.yellow_robots[0].x == 5.0
    assert result.yellow_robots[0].y == 3.5


def test_only_in_bounds_camera_contributes_to_average():
    """When two cameras see the same robot but one detection is out of bounds,
    only the in-bounds detection should contribute to the averaged position."""
    in_bounds_detection = RawRobotData(0, 1.0, 1.0, 0, 1)
    out_of_bounds_detection = RawRobotData(0, 10.0, 10.0, 0, 1)  # same id, OOB
    cam1 = RawVisionData(0, [in_bounds_detection], [], [], 0)
    cam2 = RawVisionData(0, [out_of_bounds_detection], [], [], 1)

    result = CameraCombiner().combine_cameras([cam1, cam2], tight_bounds)

    assert len(result.yellow_robots) == 1
    assert result.yellow_robots[0].x == 1.0
    assert result.yellow_robots[0].y == 1.0


if __name__ == "__main__":
    test_combine_same_robots_produces_same()
    test_combine_with_one_camera_empty()
    test_combine_with_both_camera_empty_gives_empty()
    test_combine_proximity_multiple_balls()
    test_combine_filters_low_confidence()
