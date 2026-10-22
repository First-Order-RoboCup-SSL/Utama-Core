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
    # (0, 0) and (0.02, 0.03) are one ball seen by two cameras; the others are balls of their own.
    seen = [
        (RawBallData(0, 0, 0, 1), None),
        (RawBallData(5, 5, 0, 1), None),
        (RawBallData(0.02, 0.03, 0, 1), None),
        (RawBallData(1, 1, 0, 1), None),
    ]
    balls = CameraCombiner()._combine_balls_by_proximity(seen)
    assert len(balls) == 3
    assert (balls[0].x, balls[0].y) == (0.01, 0.015)


def test_combine_with_both_camera_empty_gives_empty():
    raw_vision_data_cam2 = RawVisionData(0, [], [], [], 1)
    raw_vision_data_cam1 = RawVisionData(0, [], [], [], 0)
    combined_data = CameraCombiner().combine_cameras([raw_vision_data_cam1, raw_vision_data_cam2], infinite_bounds)

    assert combined_data.yellow_robots == []
    assert combined_data.blue_robots == []
    assert combined_data.balls == []


def test_combine_filters_low_confidence():
    seen = [
        (RawBallData(0, 0, 0, 0.05), None),
        (RawBallData(5, 5, 0, 0.05), None),
        (RawBallData(0.02, 0.02, 0, 0.05), None),
    ]
    assert len(CameraCombiner()._combine_balls_by_proximity(seen)) == 2


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


# Two cameras over the whole lab field, each behind one goal (as on 2026-10-22).
CAMERA_LEFT, CAMERA_RIGHT = (-1.73, 0.0, 2.08), (1.73, 0.0, 2.08)


def _robot_seen_at(left_xy, right_xy, left=CAMERA_LEFT, right=CAMERA_RIGHT):
    frames = [
        RawVisionData(0, [RawRobotData(2, *left_xy, 0, 1)], [], [], 0, left),
        RawVisionData(0, [RawRobotData(2, *right_xy, 0, 1)], [], [], 1, right),
    ]
    return CameraCombiner().combine_cameras(frames, infinite_bounds).yellow_robots[0]


def test_cameras_that_disagree_give_the_robot_where_the_steeper_camera_sees_it():
    # The 2026-10-22 test: at x = 0.8 the camera behind the far goal put a robot 17 cm from where
    # the camera above that end did, and the average left it 9 cm off both, jumping 9 cm whenever
    # a camera missed a frame. Beyond CAMERA_AGREEMENT (0.1 m) only the steeper camera counts.
    robot = _robot_seen_at((0.80, 0.07), (0.97, 0.13))
    assert math.isclose(robot.x, 0.97) and math.isclose(robot.y, 0.13)

    robot = _robot_seen_at((0.80, 0.07), (0.889, 0.07))  # 8.9 cm apart: both count
    assert 0.80 < robot.x < 0.889


def test_cameras_that_agree_are_weighted_towards_the_steeper_one():
    robot = _robot_seen_at((0.80, 0.0), (0.86, 0.0))
    assert robot.x > 0.83  # past the plain average, towards the camera above that end

    robot = _robot_seen_at((-0.03, 0.0), (0.03, 0.0))  # halfway between the cameras: equal weight
    assert math.isclose(robot.x, 0.0, abs_tol=1e-12)


def test_without_camera_positions_every_camera_counts_the_same():
    robot = _robot_seen_at((0.80, 0.0), (0.97, 0.0), left=None, right=None)
    assert math.isclose(robot.x, 0.885)
    robot = _robot_seen_at((0.80, 0.0), (0.97, 0.0), right=None)  # one camera unknown: no weighting
    assert math.isclose(robot.x, 0.885)


def test_robot_orientation_is_the_weighted_circular_mean():
    frames = [
        RawVisionData(0, [RawRobotData(2, 0.0, 0.0, math.pi - 0.1, 1)], [], [], 0, CAMERA_LEFT),
        RawVisionData(0, [RawRobotData(2, 0.0, 0.0, -math.pi + 0.1, 1)], [], [], 1, CAMERA_RIGHT),
    ]
    robot = CameraCombiner().combine_cameras(frames, infinite_bounds).yellow_robots[0]
    assert math.isclose(abs(robot.orientation), math.pi)


def test_a_ball_at_the_dribbler_stays_there_when_the_cameras_disagree():
    # 2026-10-22: the cameras put robot and ball 6 cm apart from each other's view. The robot was
    # averaged and the ball (beyond the old 5 cm merge) taken from one camera, so the ball came
    # out 3 cm inside the robot. Fused alike, the ball stays 11 cm ahead of the robot's centre.
    shift = 0.06
    frames = [
        RawVisionData(0, [], [RawRobotData(2, -0.9, 0.4, 0, 1)], [RawBallData(-0.79, 0.4, 0, 0.9)], 0),
        RawVisionData(0, [], [RawRobotData(2, -0.9 + shift, 0.4, 0, 1)], [RawBallData(-0.79 + shift, 0.4, 0, 0.8)], 1),
    ]
    combined = CameraCombiner().combine_cameras(frames, infinite_bounds)
    assert len(combined.balls) == 1
    assert math.isclose(combined.balls[0].x - combined.blue_robots[0].x, 0.11)

    frames[0].camera_position, frames[1].camera_position = CAMERA_LEFT, CAMERA_RIGHT  # positions known
    combined = CameraCombiner().combine_cameras(frames, infinite_bounds)
    assert len(combined.balls) == 1
    assert math.isclose(combined.balls[0].x - combined.blue_robots[0].x, 0.11, abs_tol=0.005)


def test_a_ball_only_one_camera_sees_keeps_its_place_at_the_dribbler():
    # 2026-10-22: the robot hid the ball from one camera; the robot was fused from both cameras
    # (6 cm apart) and the ball came from the other alone, so it sat 3 cm inside the robot.
    shift = 0.06
    frames = [
        RawVisionData(0, [], [RawRobotData(2, -0.9, 0.4, 0, 1)], [RawBallData(-0.79, 0.4, 0, 0.9)], 0),
        RawVisionData(0, [], [RawRobotData(2, -0.9 + shift, 0.4, 0, 1)], [], 1),
    ]
    combined = CameraCombiner().combine_cameras(frames, infinite_bounds)
    assert math.isclose(combined.balls[0].x - combined.blue_robots[0].x, 0.11)

    far = [RawVisionData(0, [], [RawRobotData(2, -0.9, 0.4, 0, 1)], [RawBallData(0.5, 0.4, 0, 0.9)], 0), frames[1]]
    assert CameraCombiner().combine_cameras(far, infinite_bounds).balls[0].x == 0.5  # no robot near: unmoved
