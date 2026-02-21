from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.data.vision import VisionBallData, VisionRobotData
from utama_core.entities.game import Ball, Field, FieldBounds, GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.run.refiners import PositionRefiner

full_field = Field.FULL_FIELD_BOUNDS
position_refiner = PositionRefiner(full_field)


def test_combining_single_team_combines_single_robot():
    zv = Vector2D(0, 0)
    game_robots = {0: Robot(0, True, False, zv, zv, zv, 0)}
    vision_robots = [VisionRobotData(0, 1, 2, 3)]
    result = position_refiner._combine_single_team_positions(game_robots, vision_robots, friendly=True)

    expected_orientation = position_refiner.angle_smoother.smooth(0, 3)

    assert len(result) == 1
    rb = result[0]
    assert rb.p.x == 1
    assert rb.p.y == 2
    assert rb.orientation == expected_orientation


def test_combining_with_robot_not_in_game_adds():
    zv = Vector2D(0, 0)
    game_robots = {0: Robot(0, True, False, zv, zv, zv, 0)}
    vision_robots = [VisionRobotData(1, 1, 2, 3)]
    result = position_refiner._combine_single_team_positions(game_robots, vision_robots, friendly=True)

    assert len(result) == 2
    rb = result[1]
    assert rb.p.x == 1
    assert rb.p.y == 2
    assert rb.orientation == 3


def test_get_most_confident_ball():
    balls = [VisionBallData(0, 0, 0, 0.2), VisionBallData(1, 2, 3, 0.8)]
    res = PositionRefiner._get_most_confident_ball(balls)

    assert res.p.x == 1
    assert res.p.y == 2
    assert res.p.z == 3


def test_most_confident_with_empty():
    res = PositionRefiner._get_most_confident_ball([])
    assert res is None


def rfac(id, is_friendly, x, y) -> Robot:
    return Robot(
        id=id,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0,
    )


def bfac(x, y) -> Ball:
    return Ball(Vector2D(x, y), Vector2D(0, 0), Vector2D(0, 0))


def base_refine(is_yellow: bool):
    friendly = {0: rfac(0, is_yellow, 0, 0)}
    enemy = {}
    # Use in-bounds positions to align with refiner's bounds filtering
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1)]
    raw_blue = [RawRobotData(0, 2, 2, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 1)
    p = PositionRefiner(full_field)
    g = GameFrame(0, is_yellow, True, friendly, enemy, bfac(0, 0))
    result = p.refine(g, [raw_vision_data_cam1, raw_vision_data_cam2])
    fr = result.friendly_robots[0]
    er = result.enemy_robots[0]

    acc_fr, acc_er = (
        raw_yellow[0] if is_yellow else raw_blue[0],
        (raw_blue[0] if is_yellow else raw_yellow[0]),
    )
    assert fr.p.x == acc_fr.x
    assert fr.p.y == acc_fr.y

    assert er.p.x == acc_er.x
    assert er.p.y == acc_er.y


def test_refine_for_yellow():
    base_refine(True)


def test_refine_for_blue():
    base_refine(False)


def test_refine_for_multiple_yellow():
    friendly = {0: rfac(0, True, 0, 0)}
    # Two in-bounds yellow robots
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1), RawRobotData(1, -2, -2, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, [], raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, raw_yellow, [], raw_balls, 1)
    p = PositionRefiner(full_field)
    g = GameFrame(0, True, True, friendly, {}, bfac(0, 0))
    result = p.refine(g, [raw_vision_data_cam1, raw_vision_data_cam2])

    assert len(result.friendly_robots) == 2
    for i in range(2):
        fr = result.friendly_robots[i]
        assert fr.p.x == raw_yellow[i].x
        assert fr.p.y == raw_yellow[i].y
        assert fr.orientation == raw_yellow[i].orientation


def test_refine_nones():
    friendly = {0: rfac(0, True, 0, 0)}
    # Two in-bounds yellow robots
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1), RawRobotData(1, -2, -2, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, [], raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, raw_yellow, [], raw_balls, 1)
    p = PositionRefiner(full_field)
    g = GameFrame(0, True, True, friendly, {}, bfac(0, 0))
    result = p.refine(g, [raw_vision_data_cam1, raw_vision_data_cam2, None, None])

    assert len(result.friendly_robots) == 2
    for i in range(2):
        fr = result.friendly_robots[i]
        assert fr.p.x == raw_yellow[i].x
        assert fr.p.y == raw_yellow[i].y
        assert fr.orientation == raw_yellow[i].orientation


def test_out_of_bounds_does_not_update_existing_robot():
    # Existing friendly robot at origin
    friendly = {0: rfac(0, True, 0, 0)}
    # Vision sees same robot far outside bounds (x beyond 5.5)
    raw_yellow = [RawRobotData(0, 10.0, 0.0, 0.0, 1.0)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, raw_yellow, [], raw_balls, 0)]

    p = PositionRefiner(full_field)
    g = GameFrame(0, True, True, friendly, {}, bfac(0, 0))
    result = p.refine(g, frames)

    # Robot should not be updated due to out-of-bounds filtering
    fr = result.friendly_robots[0]
    assert fr.p.x == 0
    assert fr.p.y == 0


def test_out_of_bounds_enemy_not_added():
    # No enemy robots initially
    friendly = {0: rfac(0, True, 0, 0)}
    # Vision sees a blue robot outside bounds (y beyond 4.0)
    raw_blue = [RawRobotData(1, 0.0, 10.0, 0.0, 1.0)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, [], raw_blue, raw_balls, 0)]

    p = PositionRefiner(full_field)
    g = GameFrame(0, True, True, friendly, {}, bfac(0, 0))
    result = p.refine(g, frames)

    # Enemy robot should not be added since it is out of bounds
    assert 1 not in result.enemy_robots


def test_out_of_bounds_friendly_not_added():
    # Vision sees a yellow robot outside bounds (y beyond 4.0)
    raw_yellow = [RawRobotData(1, 3.0, 3.1, 0.0, 1.0)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, raw_yellow, [], raw_balls, 0)]

    p = PositionRefiner(FieldBounds(top_left=(-1, 1.0), bottom_right=(1.0, -1.0)))
    g = GameFrame(0, True, True, {}, {}, bfac(0, 0))
    result = p.refine(g, frames)

    # Friendly robot should not be added since it is out of bounds
    assert 1 not in result.friendly_robots


if __name__ == "__main__":
    test_combining_single_team_combines_single_robot()
    test_combining_with_robot_not_in_game_adds()
    test_get_most_confident_ball()
    test_most_confident_with_empty()
    test_refine_for_yellow()
    test_refine_for_blue()
    test_refine_for_multiple_yellow()
    test_refine_nones()
