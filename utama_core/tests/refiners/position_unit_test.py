import math
from dataclasses import replace

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.settings import BALL_ROLL_DECEL
from utama_core.data_processing.refiners import PositionRefiner
from utama_core.data_processing.refiners.position import on_robot
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.data.vision import VisionBallData, VisionRobotData
from utama_core.entities.game import Ball, GameFrame
from utama_core.entities.game.robot import Robot

full_field_dims = STANDARD_FIELD_DIMS
position_refiner = PositionRefiner(full_field_dims)


def test_combining_single_team_combines_single_robot():
    zv = Vector2D(0, 0)
    game_robots = {0: Robot(0, True, False, zv, zv, zv, 0)}
    vision_robots = [VisionRobotData(0, 1, 2, 3)]
    result = position_refiner._combine_single_team_positions(game_robots, vision_robots, friendly=True)

    assert len(result) == 1
    rb = result[0]
    assert rb.p.x == 1
    assert rb.p.y == 2
    assert rb.orientation == 3


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


def bfac3d(x, y, z=0.0) -> Ball:
    zv = Vector3D(0, 0, 0)
    return Ball(Vector3D(x, y, z), zv, zv)


def base_refine(is_yellow: bool):
    friendly = {0: rfac(0, is_yellow, 0, 0)}
    enemy = {}
    # Use in-bounds positions to align with refiner's bounds filtering
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1)]
    raw_blue = [RawRobotData(0, 2, 2, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    raw_vision_data_cam1 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0, raw_yellow, raw_blue, raw_balls, 1)
    p = PositionRefiner(full_field_dims)
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
    p = PositionRefiner(full_field_dims)
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
    p = PositionRefiner(full_field_dims)
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
    # Vision sees same robot far outside bounds (x beyond full field)
    raw_yellow = [RawRobotData(0, 10.0, 0.0, 0.0, 1.0)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, raw_yellow, [], raw_balls, 0)]

    p = PositionRefiner(full_field_dims)
    g = GameFrame(0, True, True, friendly, {}, bfac(0, 0))
    result = p.refine(g, frames)

    # Robot should not be updated due to out-of-bounds filtering
    fr = result.friendly_robots[0]
    assert fr.p.x == 0
    assert fr.p.y == 0


def test_out_of_bounds_enemy_not_added():
    # No enemy robots initially
    friendly = {0: rfac(0, True, 0, 0)}
    # Vision sees a blue robot outside full field bounds
    raw_blue = [RawRobotData(1, 0.0, 10.0, 0.0, 1.0)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, [], raw_blue, raw_balls, 0)]

    p = PositionRefiner(full_field_dims)
    g = GameFrame(0, True, True, friendly, {}, bfac(0, 0))
    result = p.refine(g, frames)

    # Enemy robot should not be added since it is out of bounds
    assert 1 not in result.enemy_robots


def test_out_of_bounds_friendly_not_added():
    # Vision sees a yellow robot clearly outside full field bounds (+ buffer)
    raw_yellow = [RawRobotData(1, 3.0, 10.0, 0.0, 1.0)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, raw_yellow, [], raw_balls, 0)]

    p = PositionRefiner(full_field_dims)
    g = GameFrame(0, True, True, {}, {}, bfac(0, 0))
    result = p.refine(g, frames)

    # Friendly robot should not be added since it is out of bounds
    assert 1 not in result.friendly_robots


def test_allowlist_filters_stray_yellow_robot():
    # Vision sees yellow IDs 0 and 1; allowlist only permits ID 0.
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1), RawRobotData(1, -2, -2, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, raw_yellow, [], raw_balls, 0)]

    p = PositionRefiner(full_field_dims, allowed_yellow_ids=frozenset({0}))
    g = GameFrame(0, True, True, {}, {}, bfac(0, 0))
    result = p.refine(g, frames)

    assert 0 in result.friendly_robots
    assert 1 not in result.friendly_robots


def test_a_tag_off_the_robot_is_moved_back_onto_it():
    # The tag 2 cm ahead of and 1 cm left of the centre of a robot facing +y: its centre is 2 cm
    # behind (-y) and 1 cm right (+x) of the tag. Its heading stays the tag's.
    tag = RawRobotData(0, 1.0, 1.0, math.pi / 2, 1)

    robot = on_robot(tag, (0.02, 0.01))

    assert (robot.x, robot.y) == pytest.approx((1.01, 0.98))
    assert robot.orientation == pytest.approx(math.pi / 2)


def test_tag_offsets_move_only_the_robot_of_their_colour_and_id():
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1), RawRobotData(1, -2, -2, 0, 1)]
    raw_blue = [RawRobotData(0, 1, 1, 0, 1)]
    frames = [RawVisionData(0, raw_yellow, raw_blue, [RawBallData(0, 0, 0, 0)], 0)]

    p = PositionRefiner(full_field_dims, filtering=False, tag_offsets={(True, 0): (0.02, 0.0)})
    result = p.refine(GameFrame(0, True, True, {}, {}, bfac(0, 0)), frames)

    assert (result.friendly_robots[0].p.x, result.friendly_robots[0].orientation) == pytest.approx((-1.02, 0.0))
    assert (result.friendly_robots[1].p.x, result.friendly_robots[1].orientation) == (-2, 0)
    assert (result.enemy_robots[0].p.x, result.enemy_robots[0].orientation) == (1, 0)


def test_no_allowlist_passes_all_robots():
    # Without an allowlist, both IDs should be present.
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1), RawRobotData(1, -2, -2, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, raw_yellow, [], raw_balls, 0)]

    p = PositionRefiner(full_field_dims)
    g = GameFrame(0, True, True, {}, {}, bfac(0, 0))
    result = p.refine(g, frames)

    assert 0 in result.friendly_robots
    assert 1 in result.friendly_robots


def test_smooth_positions_false_passes_raw_vision_through():
    """With smooth_positions=False, present robots bypass the Kalman filter entirely and their
    vision position is used as-is."""
    friendly = {0: rfac(0, True, 0, 0)}
    raw_yellow = [RawRobotData(0, -1, -1, 0, 1)]
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, raw_yellow, [], raw_balls, 0)]

    p = PositionRefiner(full_field_dims, smooth_positions=False, impute_vanished=True)
    p.start_filtering()
    g = GameFrame(0, True, True, friendly, {}, bfac(0, 0))
    result = p.refine(g, frames)

    fr = result.friendly_robots[0]
    assert fr.p.x == raw_yellow[0].x
    assert fr.p.y == raw_yellow[0].y


def test_impute_vanished_false_leaves_vanished_robot_stale():
    """With impute_vanished=False, a robot missing from this tick's vision is neither dropped
    nor Kalman-predicted forward: it's carried over from the previous frame untouched, since
    _combine_single_team_positions only updates robots vision actually reports this tick."""
    friendly = {0: rfac(0, True, 5, 5)}
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, [], [], raw_balls, 0)]  # no yellow robots seen this tick

    p = PositionRefiner(full_field_dims, smooth_positions=True, impute_vanished=False)
    p.start_filtering()
    g = GameFrame(0, True, True, friendly, {}, bfac3d(0, 0))
    result = p.refine(g, frames)

    assert 0 in result.friendly_robots
    assert result.friendly_robots[0].p.x == 5
    assert result.friendly_robots[0].p.y == 5


def test_impute_vanished_true_keeps_vanished_robot():
    """With impute_vanished=True, a robot missing from this tick's vision is imputed via the
    Kalman predict step and remains present in the result."""
    friendly = {0: rfac(0, True, 0, 0)}
    raw_balls = [RawBallData(0, 0, 0, 0)]
    frames = [RawVisionData(0, [], [], raw_balls, 0)]  # no yellow robots seen this tick

    p = PositionRefiner(full_field_dims, smooth_positions=True, impute_vanished=True)
    p.start_filtering()
    g = GameFrame(0, True, True, friendly, {}, bfac3d(0, 0))
    result = p.refine(g, frames)

    assert 0 in result.friendly_robots


def _unseen_ball_path(v0: tuple[float, float], seconds: float, dt: float = 1 / 60) -> list[Ball]:
    """The ball, rolling at `v0` from the centre, then unseen for `seconds`. Between ticks its
    velocity is recomputed from its positions, as `VelocityRefiner` does."""
    p = PositionRefiner(full_field_dims)
    p.start_filtering()
    ball = Ball(Vector3D(0, 0, 0), Vector3D(*v0, 0), Vector3D(0, 0, 0))
    g = GameFrame(0, True, True, {}, {}, ball)
    path = []
    for k in range(1, round(seconds / dt) + 1):
        g = p.refine(g, [RawVisionData(k * dt, [], [], [], 0)])
        last = path[-1].p if path else ball.p
        v = Vector3D((g.ball.p.x - last.x) / dt, (g.ball.p.y - last.y) / dt, 0)
        g = replace(g, ball=replace(g.ball, v=v))
        path.append(g.ball)
    return path


def test_an_unseen_ball_slows_at_the_rolling_deceleration_and_stops():
    """The rsim ball 'escape' (2026-10-21): a ball unseen kept its last velocity forever and was
    600 m away after 15 minutes. Unseen, it now rolls as a real ball does, then stops."""
    path = _unseen_ball_path((1.0, 0.0), seconds=5.0)
    stop = 1.0 / BALL_ROLL_DECEL
    assert path[59].p.x == pytest.approx(1.0 - 0.5 * BALL_ROLL_DECEL, abs=0.01)  # 1 s in
    assert path[-1].p.x == pytest.approx(0.5 * stop, abs=0.01)  # 1 m/s rolls 1/(2a) m, then stops
    assert path[-1].p.y == 0


def test_an_unseen_ball_stays_where_vision_would_still_see_it():
    """Fast and unseen towards the corner: it stops at the edge of what vision accepts (field plus
    `VISION_BOUNDS_BUFFER`), not beyond."""
    path = _unseen_ball_path((4.0, 4.0), seconds=10.0)
    bounds = PositionRefiner(full_field_dims).vision_bounds
    assert (path[-1].p.x, path[-1].p.y) == pytest.approx((bounds.x_max, bounds.y_max))
    assert max(abs(b.p.x) for b in path) <= bounds.x_max


def test_with_smoothing_off_a_second_loss_starts_where_the_ball_was_last_seen():
    """Smoothing off, imputation on: the filter only runs while the ball is unseen, so it kept
    the first loss and a second one carried on that roll from where the first started."""
    p = PositionRefiner(full_field_dims, smooth_positions=False, impute_vanished=True)
    p.start_filtering()
    g = GameFrame(0, True, True, {}, {}, Ball(Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(0, 0, 0)))
    g = p.refine(g, [RawVisionData(0.1, [], [], [], 0)])  # lost, rolling on at 1 m/s
    g = replace(g, ball=replace(g.ball, v=Vector3D(1, 0, 0)))
    g = p.refine(g, [RawVisionData(0.2, [], [], [RawBallData(-1.0, 0.5, 0, 1)], 0)])  # seen, still
    assert (g.ball.p.x, g.ball.p.y) == pytest.approx((-1.0, 0.5))
    g = replace(g, ball=replace(g.ball, v=Vector3D(0, 0, 0)))
    g = p.refine(g, [RawVisionData(0.3, [], [], [], 0)])  # lost again
    assert (g.ball.p.x, g.ball.p.y) == pytest.approx((-1.0, 0.5), abs=0.01)


def test_with_no_filtering_a_seen_ball_passes_through():
    p = PositionRefiner(full_field_dims, filtering=False)
    p.start_filtering()
    g = GameFrame(0, True, True, {}, {}, Ball(Vector3D(0, 0, 0), Vector3D(0, 0, 0), Vector3D(0, 0, 0)))
    g = p.refine(g, [RawVisionData(0.1, [], [], [RawBallData(-1.0, 0.5, 0, 1)], 0)])
    assert (g.ball.p.x, g.ball.p.y) == pytest.approx((-1.0, 0.5))


def test_the_refiner_keeps_a_cameras_gap_between_frames():
    # The combiner remembers each camera's gap to the others at a robot; a fresh combiner per
    # frame would forget it, and a ball whose camera just lost the robot lands inside the robot.
    p = PositionRefiner(full_field_dims, filtering=False)
    g = GameFrame(0, True, True, {}, {}, bfac3d(0, 0))
    for ts, robot_1 in ((0.0, [RawRobotData(2, -0.415, 0.3, 0, 1)]), (0.016, [])):
        frames = [
            RawVisionData(ts, [], [RawRobotData(2, -0.5, 0.3, 0, 1)], [], 0),
            RawVisionData(ts, [], robot_1, [RawBallData(-0.3, 0.3, 0, 0.9)], 1),
        ]
        g = p.refine(g, frames)
    assert g.ball.p.x - g.enemy_robots[2].p.x == pytest.approx(0.115)


def _picked(p, last, balls):
    g = GameFrame(0, True, True, {}, {}, last)
    return p.refine(g, [RawVisionData(0.1, [], [], balls, 0)]).ball


def test_stray_balls_off_the_field_do_not_take_the_ball():
    # 2026-10-22: camera 0 also saw four steady "balls" 15 cm past the +y touch line, as confident
    # as the real one; the most confident, ties in list order, picked a fake 22% of the time.
    p = PositionRefiner(full_field_dims, filtering=False)
    w = full_field_dims.full_field_half_width
    fakes = [RawBallData(x, w + 0.15, 0, 1.0) for x in (-0.705, -0.22, -0.075, 0.079)]
    real = RawBallData(-0.8, -0.35, 0, 1.0)
    for balls in (fakes + [real], [real] + fakes, fakes[:2] + [real] + fakes[2:]):
        ball = _picked(p, bfac3d(-0.8, -0.35), balls)
        assert (ball.p.x, ball.p.y) == pytest.approx((-0.8, -0.35))
    alone = _picked(p, bfac3d(0, 0), [fakes[0]])  # the only ball: taken, though off the field
    assert alone.p.y == pytest.approx(w + 0.15)


def test_among_balls_in_play_the_one_nearest_the_last_is_kept():
    p = PositionRefiner(full_field_dims, filtering=False)
    balls = [RawBallData(1.0, 0.0, 0, 1.0), RawBallData(-1.0, 0.0, 0, 0.5)]
    assert _picked(p, bfac3d(-0.9, 0.0), balls).p.x == pytest.approx(-1.0)
    assert _picked(p, None, balls).p.x == pytest.approx(1.0)  # no last ball: the most confident
    # A ball in the goal is in play (2026-10-22: a shot past the goal line, a blob elsewhere).
    in_goal = full_field_dims.full_field_half_length + 0.05
    balls = [RawBallData(1.0, 0.0, 0, 1.0), RawBallData(in_goal, 0.0, 0, 1.0)]
    assert _picked(p, bfac3d(in_goal, 0.0), balls).p.x == pytest.approx(in_goal)


if __name__ == "__main__":
    test_combining_single_team_combines_single_robot()
    test_combining_with_robot_not_in_game_adds()
    test_get_most_confident_ball()
    test_most_confident_with_empty()
    test_refine_for_yellow()
    test_refine_for_blue()
    test_refine_for_multiple_yellow()
    test_refine_nones()
