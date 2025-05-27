import vector
import numpy as np
from vector import VectorObject3D
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.past_game import PastGame, TeamType, get_structured_object_key
from entities.game.robot import Robot
from run.refiners import VelocityRefiner
import pytest

velocity_refiner = VelocityRefiner()


def create_ball_only_game(
    ts: float, x: float, y: float, z: float, vx: float = 1, vy: float = 1, vz: float = 1
):
    return Game(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={},
        enemy_robots={},
        ball=Ball(vector.obj(x=x, y=y, z=z), vector.obj(x=vx, y=vy, z=vz), None),
    )


def create_one_robot_only_game(ts: float, x: float, y: float, is_friendly: bool):
    robots = {
        1: Robot(
            id=1,
            is_friendly=is_friendly,
            has_ball=True,
            p=vector.obj(x=x, y=y),
            v=None,
            a=None,
            orientation=0,
        )
    }
    return Game(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={} if not is_friendly else robots,
        enemy_robots={} if is_friendly else robots,
        ball=None,
    )


def test_velocity_calculation_correct_for_ball():
    past_game = PastGame(10)
    past_game.add_game(create_ball_only_game(2, 10, 20, 30))

    game = create_ball_only_game(5, 19, 32, 33)
    velocity_refiner = VelocityRefiner()

    game = velocity_refiner.refine(past_game, game)

    assert game.ball.v.x == 3
    assert game.ball.v.y == 4
    assert game.ball.v.z == 1


@pytest.mark.parametrize("is_friendly", [True, False])
def test_velocity_calculation_correct_for_one_robot(is_friendly):
    past_game = PastGame(10)
    past_game.add_game(create_one_robot_only_game(4, 15, 30, is_friendly))
    game = create_one_robot_only_game(10, 15, 3, is_friendly)
    game = velocity_refiner.refine(past_game, game)

    target_robot = game.enemy_robots[1] if not is_friendly else game.friendly_robots[1]

    assert target_robot.v.x == 0
    assert target_robot.v.y == -4.5


def test_returns_not_moving_if_not_enough_information_for_velocity():
    past_game = PastGame(10)
    game = create_ball_only_game(1, 1, 1, 1)
    game = velocity_refiner.refine(past_game, game)
    assert game.ball.v.x == 0
    assert game.ball.v.y == 0


def test_extraction_of_time_velocity_pairs():
    past_game = PastGame(20)
    all_added_game_data = []

    for i in range(20):
        time = float(i)
        vx, vy, vz = 1.0, 1.0, 1.0
        game_to_add = create_ball_only_game(time, time, time, time, vx, vy, vz)
        past_game.add_game(game_to_add)
        all_added_game_data.append((time, vector.obj(x=vx, y=vy, z=vz)))

    points_needed = (
        VelocityRefiner.ACCELERATION_N_WINDOWS
        * VelocityRefiner.ACCELERATION_WINDOW_SIZE
    )  # e.g., 15

    if len(all_added_game_data) >= points_needed:
        start_index_for_expected = len(all_added_game_data) - points_needed
        expected_time_velocity_pairs = all_added_game_data[start_index_for_expected:]
    elif len(all_added_game_data) > 0:
        expected_time_velocity_pairs = all_added_game_data[:]
    else:
        expected_time_velocity_pairs = []

    game_for_extraction_call = create_ball_only_game(0, 0, 0, 0)
    ball_obj_key = get_structured_object_key(
        game_for_extraction_call.ball, TeamType.NEUTRAL
    )

    extracted_ts_np, extracted_vel_np = (
        velocity_refiner._extract_time_velocity_np_arrays(
            past_game, ball_obj_key, points_needed
        )
    )

    assert len(extracted_ts_np) == len(extracted_vel_np)

    reconstructed_extracted_pairs = []
    if extracted_ts_np.ndim > 0 and extracted_vel_np.ndim > 0:
        for i in range(len(extracted_ts_np)):
            ts = extracted_ts_np[i]
            vel_components = extracted_vel_np[i]
            vec_obj = VectorObject3D(
                x=vel_components[0], y=vel_components[1], z=vel_components[2]
            )
            reconstructed_extracted_pairs.append((ts, vec_obj))

    assert len(reconstructed_extracted_pairs) == len(
        expected_time_velocity_pairs
    ), "Number of items mismatch"

    for actual_pair, expected_pair in zip(
        reconstructed_extracted_pairs, expected_time_velocity_pairs
    ):
        actual_ts, actual_vel = actual_pair
        expected_ts, expected_vel = expected_pair

        assert np.isclose(
            actual_ts, expected_ts
        ), f"Timestamp mismatch: actual {actual_ts}, expected {expected_ts}"
        assert (
            actual_vel == expected_vel
        ), f"Velocity mismatch: actual {actual_vel}, expected {expected_vel}"


def test_acceleration_calculation_implements_expected_formula():
    past_game = PastGame(20)
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
            past_game.add_game(
                create_ball_only_game(
                    ts, 0, 0, 0, acc * i + noise, acc * i + noise, acc * i + noise
                )
            )
            ts += 1

    game = create_ball_only_game(ts, 0, 0, 0)
    game = velocity_refiner.refine(past_game, game)

    assert game.ball.a.x == pytest.approx(
        acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE
    )
    assert game.ball.a.y == pytest.approx(
        acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE
    )
    assert game.ball.a.z == pytest.approx(
        acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE
    )
