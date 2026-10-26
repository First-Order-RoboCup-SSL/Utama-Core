"""Local camera consistency: one physical object, one contribution per camera."""

import json
import math
from pathlib import Path

import pytest

from utama_core.config.field_params import FieldDimensions
from utama_core.data_processing.refiners import PositionRefiner, VelocityRefiner
from utama_core.data_processing.refiners.position import CameraCombiner, VisionBounds
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.game import GameFrame, GameHistory

BOUNDS = VisionBounds(-10, 10, -10, 10)


def test_nearby_ball_duplicates_use_the_strongest_detection_per_camera():
    frames = [
        RawVisionData(1, [], [], [RawBallData(0, 0, 0, 0.4), RawBallData(0.02, 0, 0, 0.95)], 0),
        RawVisionData(1, [], [], [RawBallData(0.1, 0, 0, 1)], 1),
    ]
    result = CameraCombiner().combine_cameras(frames, BOUNDS)
    assert len(result.balls) == 1
    assert result.balls[0].x == pytest.approx(0.06)  # camera 0 winner .02; camera 1 .10


def test_nearby_same_id_robot_duplicates_do_not_give_one_camera_extra_weight():
    frames = [
        RawVisionData(1, [], [RawRobotData(2, 0, 0, 0, 1), RawRobotData(2, 0.02, 0, 0, 1)], [], 0),
        RawVisionData(1, [], [RawRobotData(2, 0.1, 0, 0, 1)], [], 1),
    ]
    result = CameraCombiner().combine_cameras(frames, BOUNDS)
    assert len(result.blue_robots) == 1
    assert result.blue_robots[0].x == pytest.approx(0.055)


@pytest.mark.parametrize("colour", ["yellow", "blue"])
@pytest.mark.parametrize("axis,sign", [("x", -1), ("x", 1), ("y", -1), ("y", 1)])
@pytest.mark.parametrize("inside", [2.45, 2.49, 2.5])
def test_outside_robot_repeat_cannot_erase_or_bias_an_inside_observation(colour, axis, sign, inside):
    bounds = VisionBounds(-2.5, 2.5, -2.5, 2.5)
    x, y = (sign * inside, 0) if axis == "x" else (0, sign * inside)
    outside_x, outside_y = (sign * 2.55, 0) if axis == "x" else (0, sign * 2.55)
    robots = [RawRobotData(2, x, y, 0.4, 0.9), RawRobotData(2, outside_x, outside_y, 1, 0.99)]
    frame = RawVisionData(1, robots if colour == "yellow" else [], robots if colour == "blue" else [], [], 0)

    result = CameraCombiner().combine_cameras([frame], bounds)

    combined = result.yellow_robots if colour == "yellow" else result.blue_robots
    assert len(combined) == 1
    assert combined[0].id == 2
    assert (combined[0].x, combined[0].y, combined[0].orientation) == pytest.approx((x, y, 0.4))


def test_close_different_robot_identities_are_not_silently_combined():
    frame = RawVisionData(1, [], [RawRobotData(2, 0, 0, 0, 1), RawRobotData(3, 0.08, 0, 0, 1)], [], 0)
    assert {r.id for r in CameraCombiner().combine_cameras([frame], BOUNDS).blue_robots} == {2, 3}


def test_real_single_camera_ball_does_not_move_inside_robot_after_filtering():
    # Four minimized control inputs from 20261022_211142; only the far camera sees the ball.
    steps = json.loads((Path(__file__).with_name("one_camera_ball_repro.json")).read_text())
    position = PositionRefiner(
        FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4),
        allowed_blue_ids=frozenset({2}),
        allowed_yellow_ids=frozenset(),
    )
    velocity = VelocityRefiner()
    history = GameHistory(20)
    current = GameFrame(0, False, False, {}, {}, None)
    for step in steps:
        frames = [
            RawVisionData(
                s["ts"],
                [RawRobotData(**r) for r in s["yellow_robots"]],
                [RawRobotData(**r) for r in s["blue_robots"]],
                [RawBallData(**b) for b in s["balls"]],
                s["camera_id"],
                s["camera_position"],
            )
            for s in step
        ]
        new = position.refine(current, frames)
        if position.filter_running:
            new = velocity.refine(history, new)
        elif new.ball is not None:
            position.start_filtering()
        if current.ts:
            history.add_game_frame(current)
        current = new
    ball = current.ball.p
    robot = current.friendly_robots[2].p
    assert math.hypot(ball.x - robot.x, ball.y - robot.y) >= 0.09
    # The Kalman state must follow the correction rather than springing back next tick.
    assert tuple(position.kalman_filter_ball.state[:2]) == pytest.approx((ball.x, ball.y))


def test_ignored_nearby_robot_id_cannot_displace_the_ball_from_active_robot():
    position = PositionRefiner(
        FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4),
        filtering=False,
        allowed_blue_ids=frozenset({2}),
        allowed_yellow_ids=frozenset(),
    )
    frames = [
        RawVisionData(
            1, [], [RawRobotData(2, 0, 0, 0, 1), RawRobotData(3, 0.01, 0, 0, 1)], [RawBallData(0.115, 0, 0, 1)], 0
        ),
        RawVisionData(1, [], [RawRobotData(2, 0.08, 0, 0, 1)], [], 1),
    ]
    result = position.refine(GameFrame(0, False, False, {}, {}, None), frames)
    assert set(result.friendly_robots) == {2}
    assert result.ball.p.x - result.friendly_robots[2].p.x == pytest.approx(0.115)


def test_far_camera_ball_keeps_offset_when_nearer_camera_dominates_robot_pose():
    frames = [
        RawVisionData(1, [], [RawRobotData(2, 0.8, 0, 0, 1)], [RawBallData(0.915, 0, 0, 1)], 0, (-1.73, 0, 2.08)),
        RawVisionData(1, [], [RawRobotData(2, 0.885, 0, 0, 1)], [], 1, (1.73, 0, 2.08)),
    ]
    result = CameraCombiner().combine_cameras(frames, BOUNDS)
    assert result.blue_robots[0].x > 0.8425
    assert result.balls[0].x - result.blue_robots[0].x == pytest.approx(0.115)
    assert frames[0].balls[0].x == pytest.approx(0.915)  # original camera observation stays intact


@pytest.mark.parametrize("reverse", [False, True])
def test_a_stronger_distant_candidate_cannot_displace_the_nearby_track(reverse):
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=False)
    start = RawVisionData(1, [], [], [RawBallData(0, 0, 0, 0.8)], 0)
    current = position.refine(GameFrame(0, False, False, {}, {}, None), [start])
    candidates = [RawBallData(0.02, 0, 0, 0.8), RawBallData(0.04, 0, 0, 0.9), RawBallData(1.4, 0, 0, 1)]
    if reverse:
        candidates.reverse()
    frame = RawVisionData(1.02, [], [], candidates, 0)
    result = position.refine(current, [frame])
    assert result.ball.p.x == pytest.approx(0.04)
    assert len(frame.balls) == 3  # do not alter original camera observations


def test_a_chain_of_nearby_candidates_cannot_merge_two_distant_endpoints():
    frames = [RawVisionData(1, [], [], [RawBallData(x, 0, 0, 1)], i) for i, x in enumerate((0, -0.19, 0.19))]
    balls = CameraCombiner().combine_cameras(frames, BOUNDS).balls
    assert len(balls) == 2  # endpoints are 380 mm apart, beyond the 200 mm merge limit


def test_a_camera_that_loses_the_ball_stops_contributing_to_fusion():
    combiner = CameraCombiner()
    combiner.combine_cameras(
        [
            RawVisionData(1, [], [], [RawBallData(0, 0, 0, 1)], 0),
            RawVisionData(1, [], [], [RawBallData(0.1, 0, 0, 1)], 1),
        ],
        BOUNDS,
    )
    result = combiner.combine_cameras(
        [
            RawVisionData(1.02, [], [], [], 0),
            RawVisionData(1.02, [], [], [RawBallData(0.12, 0, 0, 1)], 1),
        ],
        BOUNDS,
    )
    assert result.balls[0].x == pytest.approx(0.12)
    missing = combiner.combine_cameras([RawVisionData(1.04, [], [], [], 0), RawVisionData(1.04, [], [], [], 1)], BOUNDS)
    assert missing.balls == []


@pytest.mark.parametrize("filtering", [False, True])
def test_strategy_keeps_a_ball_estimate_when_both_cameras_lose_it(filtering):
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=filtering)
    start = RawVisionData(1, [], [], [RawBallData(0.4, 0.2, 0, 1)], 0)
    current = position.refine(GameFrame(0, False, False, {}, {}, None), [start])
    position.start_filtering()
    for ts in (1.02, 1.04, 2):
        result = position.refine(current, [RawVisionData(ts, [], [], [], 0), RawVisionData(ts, [], [], [], 1)])
        assert result.ball is not None
        assert (result.ball.p.x, result.ball.p.y) == pytest.approx((0.4, 0.2))
        current = result
    assert position.refine(current, [None, None]).ball is current.ball


@pytest.mark.parametrize("distance,count", [(0.1999, 1), (0.2, 2), (0.2001, 2)])
def test_ball_merge_limit_is_strictly_200mm(distance, count):
    frames = [
        RawVisionData(1, [], [], [RawBallData(0, 0, 0, 1)], 0),
        RawVisionData(1, [], [], [RawBallData(distance, 0, 0, 1)], 1),
    ]
    assert len(CameraCombiner().combine_cameras(frames, BOUNDS).balls) == count


@pytest.mark.parametrize("filtering", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cameras", [1, 2])
def test_robot_pattern_cannot_replace_a_plausible_nearby_ball(filtering, reverse, cameras):
    position = PositionRefiner(
        FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=filtering, allowed_blue_ids=frozenset({2})
    )

    def frames(ts, candidates):
        result = [RawVisionData(ts, [], [RawRobotData(2, 0, 0, 0, 1)], candidates, 0)]
        if cameras == 2:
            result.append(RawVisionData(ts, [], [RawRobotData(2, 0.08, 0, 0, 1)], [], 1))
        return result

    current = position.refine(GameFrame(0, False, False, {}, {}, None), frames(1, [RawBallData(0.115, 0, 0, 0.9)]))
    position.start_filtering()
    candidates = [RawBallData(0.115, 0, 0, 0.9), RawBallData(0, 0, 0, 0.99)]
    if reverse:
        candidates.reverse()
    observation = frames(1.02, candidates)
    result = position.refine(current, observation)
    assert result.ball.p.x - result.friendly_robots[2].p.x == pytest.approx(0.115)
    assert observation[0].balls == candidates  # raw observations remain intact


@pytest.mark.parametrize("filtering", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_in_field_ball_is_not_discarded_for_a_stronger_nearby_outside_candidate(filtering, reverse):
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=filtering)
    current = position.refine(
        GameFrame(0, False, False, {}, {}, None), [RawVisionData(1, [], [], [RawBallData(1.4, 0.8, 0, 0.9)], 0)]
    )
    position.start_filtering()
    candidates = [RawBallData(1.4, 0.8, 0, 0.9), RawBallData(1.55, 0.8, 0, 0.99)]
    if reverse:
        candidates.reverse()
    result = position.refine(current, [RawVisionData(1.02, [], [], candidates, 0)])
    assert result.ball.p.x == pytest.approx(1.4)


@pytest.mark.parametrize("filtering", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_reset_preserves_preference_for_an_in_field_ball(filtering, reverse):
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=filtering)
    candidates = [RawBallData(1.4, 0.8, 0, 0.9), RawBallData(1.55, 0.8, 0, 0.99)]
    if reverse:
        candidates.reverse()

    for _ in range(3):
        current = position.refine(GameFrame(0, False, False, {}, {}, None), [RawVisionData(1, [], [], candidates, 0)])
        assert (current.ball.p.x, current.ball.p.y) == pytest.approx((1.4, 0.8))
        position.start_filtering()
        result = position.refine(current, [RawVisionData(1.02, [], [], candidates, 0)])
        assert (result.ball.p.x, result.ball.p.y) == pytest.approx((1.4, 0.8))
        position.reset()


def test_a_ball_inside_the_goal_is_a_plausible_candidate():
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=False)
    candidates = [RawBallData(1.4, 0, 0, 0.9), RawBallData(1.55, 0, 0, 0.99)]
    result = position.refine(GameFrame(0, False, False, {}, {}, None), [RawVisionData(1, [], [], candidates, 0)])
    assert result.ball.p.x == pytest.approx(1.55)


def test_a_raised_ball_is_not_rejected_by_a_flat_robot_footprint():
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=False)
    frame = RawVisionData(
        1, [], [RawRobotData(2, 0, 0, 0, 1)], [RawBallData(0.115, 0, 0, 0.9), RawBallData(0, 0, 0.2, 0.99)], 0
    )
    result = position.refine(GameFrame(0, False, False, {}, {}, None), [frame])
    assert (result.ball.p.x, result.ball.p.z) == pytest.approx((0, 0.2))


def test_a_single_uncertain_ball_measurement_is_not_erased_by_a_shape_proxy():
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=False)
    frame = RawVisionData(1, [], [RawRobotData(2, 0, 0, 0, 1)], [RawBallData(0.075, 0, 0, 0.9)], 0)
    result = position.refine(GameFrame(0, False, False, {}, {}, None), [frame])
    assert result.ball.p.x == pytest.approx(0.075)


def test_a_second_camera_robot_pattern_cannot_pull_a_valid_ball_inside_the_robot():
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=False)
    frames = [
        RawVisionData(1, [], [RawRobotData(2, 0, 0, 0, 1)], [RawBallData(0.115, 0, 0, 0.9)], 0),
        RawVisionData(1, [], [RawRobotData(2, 0.08, 0, 0, 1)], [RawBallData(0.08, 0, 0, 0.99)], 1),
    ]
    result = position.refine(GameFrame(0, False, False, {}, {}, None), frames)
    assert result.ball.p.x - result.friendly_robots[2].p.x == pytest.approx(0.115)


def test_a_second_camera_outside_candidate_cannot_pull_a_valid_ball_out_of_field():
    position = PositionRefiner(FieldDimensions(1.5, 1.125, 0.25, 0.5, 0.4, 0.4), filtering=False)
    frames = [
        RawVisionData(1, [], [], [RawBallData(1.4, 0.8, 0, 0.9)], 0),
        RawVisionData(1, [], [], [RawBallData(1.55, 0.8, 0, 0.99)], 1),
    ]
    result = position.refine(GameFrame(0, False, False, {}, {}, None), frames)
    assert result.ball.p.x == pytest.approx(1.4)
