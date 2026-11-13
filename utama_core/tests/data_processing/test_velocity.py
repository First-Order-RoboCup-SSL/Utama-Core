import numpy as np
import pytest

from utama_core.data_processing.refiners import VelocityRefiner
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import (
    GameHistory,
    TeamType,
    get_structured_object_key,
)
from utama_core.entities.game.robot import Robot

velocity_refiner = VelocityRefiner()


def create_ball_only_game(ts: float, x: float, y: float, z: float, vx: float = 1, vy: float = 1, vz: float = 1):
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={},
        enemy_robots={},
        ball=Ball(Vector3D(x, y, z), Vector3D(vx, vy, vz), None),
    )


def create_one_robot_only_game(ts: float, x: float, y: float, is_friendly: bool):
    robots = {
        1: Robot(
            id=1,
            is_friendly=is_friendly,
            has_ball=True,
            p=Vector2D(x, y),
            v=None,
            a=None,
            orientation=0,
        )
    }
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={} if not is_friendly else robots,
        enemy_robots={} if is_friendly else robots,
        ball=None,
    )


def test_velocity_calculation_correct_for_ball():
    game_history = GameHistory(10)
    game_history.add_game_frame(create_ball_only_game(2, 10, 20, 30))

    game = create_ball_only_game(5, 19, 32, 33)
    velocity_refiner = VelocityRefiner()

    game = velocity_refiner.refine(game_history, game)

    assert game.ball.v.x == 3
    assert game.ball.v.y == 4
    assert game.ball.v.z == 1


@pytest.mark.parametrize("is_friendly", [True, False])
def test_velocity_calculation_correct_for_one_robot(is_friendly):
    game_history = GameHistory(10)
    game_history.add_game_frame(create_one_robot_only_game(4, 15, 30, is_friendly))
    game = create_one_robot_only_game(10, 15, 21, is_friendly)
    game = velocity_refiner.refine(game_history, game)

    target_robot = game.enemy_robots[1] if not is_friendly else game.friendly_robots[1]

    assert target_robot.v.x == 0
    assert target_robot.v.y == -1.5


def test_returns_not_moving_if_not_enough_information_for_velocity():
    game_history = GameHistory(10)
    game = create_ball_only_game(1, 1, 1, 1)
    game = velocity_refiner.refine(game_history, game)
    assert game.ball.v.x == 0
    assert game.ball.v.y == 0


def test_extraction_of_time_velocity_pairs():
    game_history = GameHistory(20)
    all_added_game_data = []

    for i in range(20):
        time = float(i)
        vx, vy, vz = 1.0, 1.0, 1.0
        game_to_add = create_ball_only_game(time, time, time, time, vx, vy, vz)
        game_history.add_game_frame(game_to_add)
        all_added_game_data.append((time, Vector3D(vx, vy, vz)))

    points_needed = VelocityRefiner.ACCELERATION_N_WINDOWS * VelocityRefiner.ACCELERATION_WINDOW_SIZE  # e.g., 15

    if len(all_added_game_data) >= points_needed:
        start_index_for_expected = len(all_added_game_data) - points_needed
        expected_time_velocity_pairs = all_added_game_data[start_index_for_expected:]
    elif len(all_added_game_data) > 0:
        expected_time_velocity_pairs = all_added_game_data[:]
    else:
        expected_time_velocity_pairs = []

    game_for_extraction_call = create_ball_only_game(0, 0, 0, 0)
    ball_obj_key = get_structured_object_key(game_for_extraction_call.ball, TeamType.NEUTRAL)

    extracted_ts_np, extracted_vel_np = velocity_refiner._extract_time_velocity_np_arrays(
        game_history, ball_obj_key, points_needed
    )

    assert len(extracted_ts_np) == len(extracted_vel_np)

    reconstructed_extracted_pairs = []
    if extracted_ts_np.ndim > 0 and extracted_vel_np.ndim > 0:
        for i in range(len(extracted_ts_np)):
            ts = extracted_ts_np[i]
            vel_components = extracted_vel_np[i]
            vec_obj = Vector3D(vel_components[0], vel_components[1], vel_components[2])
            reconstructed_extracted_pairs.append((ts, vec_obj))

    assert len(reconstructed_extracted_pairs) == len(expected_time_velocity_pairs), "Number of items mismatch"

    for actual_pair, expected_pair in zip(reconstructed_extracted_pairs, expected_time_velocity_pairs):
        actual_ts, actual_vel = actual_pair
        expected_ts, expected_vel = expected_pair

        assert np.isclose(actual_ts, expected_ts), f"Timestamp mismatch: actual {actual_ts}, expected {expected_ts}"
        assert actual_vel == expected_vel, f"Velocity mismatch: actual {actual_vel}, expected {expected_vel}"


def test_acceleration_calculation_implements_expected_formula():
    game_history = GameHistory(20)
    acc = 3.6

    ts = 0
    for i in range(VelocityRefiner.ACCELERATION_N_WINDOWS):
        for j in range(VelocityRefiner.ACCELERATION_WINDOW_SIZE):
            if j == 0:
                noise = 100
            elif j == 1:
                noise = -100
            else:
                noise = 0
            game_history.add_game_frame(
                create_ball_only_game(ts, 0, 0, 0, acc * i + noise, acc * i + noise, acc * i + noise)
            )
            ts += 1

    game = create_ball_only_game(ts, 0, 0, 0)
    game = velocity_refiner.refine(game_history, game)

    assert game.ball.a.x == pytest.approx(acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE)
    assert game.ball.a.y == pytest.approx(acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE)
    assert game.ball.a.z == pytest.approx(acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE)


def test_acceleration_left_zero_when_not_computed():
    game_history = GameHistory(20)
    for ts in range(VelocityRefiner.ACCELERATION_N_WINDOWS * VelocityRefiner.ACCELERATION_WINDOW_SIZE):
        game_history.add_game_frame(create_ball_only_game(ts, ts, 0, 0, ts, ts, ts))

    game = VelocityRefiner(compute_acceleration=False).refine(game_history, create_ball_only_game(15, 15, 0, 0))

    assert game.ball.v.x == pytest.approx(1)
    assert (game.ball.a.x, game.ball.a.y, game.ball.a.z) == (0, 0, 0)


def _velocity_entries(dts, velocities):
    ts, entries = 10.0, []
    for dt, v in zip(dts, velocities):
        ts += dt
        entries.append((ts, v))
    return entries


@pytest.mark.parametrize(
    "dts, velocities",
    [
        ([1 / 60] * 15, [(0.1 * i, -0.05 * i * i) for i in range(15)]),  # accelerating
        ([1 / 60] * 5 + [0.0] * 5 + [1 / 60] * 5, [(1.0, 2.0)] * 7 + [(1.5, -0.0)] * 8),  # one zero-dt segment
        ([0.0] * 15, [(1.0, 1.0)] * 15),  # every segment too short: zero, with a warning
        ([1 / 60] * 15, [(-0.0, 0.0)] * 15),  # signed zeros keep their sign
    ],
)
def test_robot_acceleration_fast_path_is_the_general_helper_bit_for_bit(dts, velocities):
    # Robots read their velocity history in place instead of through
    # `_windowed_average_derivative`: the same additions in the same order, so the same floats.
    entries = _velocity_entries(dts, velocities)
    general = velocity_refiner._windowed_average_derivative(
        [float(ts) for ts, _ in entries],
        [vec for _, vec in entries],
        VelocityRefiner.ACCELERATION_N_WINDOWS,
        VelocityRefiner.ACCELERATION_WINDOW_SIZE,
        True,
        log_prefix="ACCEL",
    )
    fast = velocity_refiner._acceleration_from_entries(entries, True)
    assert (repr(fast.x), repr(fast.y)) == (repr(general.x), repr(general.y))


def test_velocity_skips_a_stored_position_too_close_in_time():
    """2026-10-27 field run: two cameras' frames 0.01 ms apart, 1 mm apart in position, gave
    64 m/s. Differenced instead against the newest position at least MIN_VELOCITY_DT_S older."""
    game_history = GameHistory(10)
    game_history.add_game_frame(create_one_robot_only_game(1.0, 0.0, 0.0, is_friendly=True))
    game_history.add_game_frame(create_one_robot_only_game(1.02 - 1e-5, 0.020, 0.0, is_friendly=True))

    game = velocity_refiner.refine(game_history, create_one_robot_only_game(1.02, 0.021, 0.0, is_friendly=True))

    assert game.friendly_robots[1].v.x == pytest.approx(0.021 / 0.02)


@pytest.mark.parametrize(
    "gap, uses_newest",
    [(VelocityRefiner.MIN_VELOCITY_DT_S + 1e-9, True), (VelocityRefiner.MIN_VELOCITY_DT_S - 1e-4, False)],
)
def test_velocity_uses_the_newest_position_from_min_dt_old(gap, uses_newest):
    game_history = GameHistory(10)
    game_history.add_game_frame(create_one_robot_only_game(0.5, 0.0, 0.0, is_friendly=True))
    game_history.add_game_frame(create_one_robot_only_game(1.0 - gap, 0.5, 0.0, is_friendly=True))

    game = velocity_refiner.refine(game_history, create_one_robot_only_game(1.0, 0.5 + gap, 0.0, is_friendly=True))

    expected = 1.0 if uses_newest else (0.5 + gap) / 0.5
    assert game.friendly_robots[1].v.x == pytest.approx(expected)


def test_velocity_is_zero_when_no_stored_position_is_old_enough():
    game_history = GameHistory(10)
    game_history.add_game_frame(create_one_robot_only_game(1.0, 0.0, 0.0, is_friendly=True))

    game = velocity_refiner.refine(game_history, create_one_robot_only_game(1.0 + 1e-5, 0.001, 0.0, is_friendly=True))

    assert game.friendly_robots[1].v.x == 0.0


@pytest.mark.parametrize("step_m, expected", [(0.04, 2.0), (0.08, VelocityRefiner.MAX_ROBOT_SPEED_MPS)])
def test_a_robot_velocity_is_capped_and_keeps_its_direction(step_m, expected):
    """A Kalman snap after a dropout read 20-380 m/s on the field (2026-10-27); a robot's speed
    is capped at MAX_ROBOT_SPEED_MPS, along the same direction. A ball's is not capped."""
    game_history = GameHistory(10)
    game_history.add_game_frame(create_one_robot_only_game(1.0, 0.0, 0.0, is_friendly=False))
    frame = create_one_robot_only_game(1.02, 0.6 * step_m, 0.8 * step_m, is_friendly=False)

    v = velocity_refiner.refine(game_history, frame).enemy_robots[1].v

    assert v.mag() == pytest.approx(expected)
    assert v.y / v.x == pytest.approx(0.8 / 0.6)


def test_a_ball_velocity_is_not_capped():
    game_history = GameHistory(10)
    game_history.add_game_frame(create_ball_only_game(1.0, 0.0, 0.0, 0.0))

    game = velocity_refiner.refine(game_history, create_ball_only_game(1.02, 0.12, 0.0, 0.0))

    assert game.ball.v.x == pytest.approx(6.0)
