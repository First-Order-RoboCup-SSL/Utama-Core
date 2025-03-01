import vector
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.past_game import PastGame
from entities.game.robot import Robot
from refiners.velocity import VelocityRefiner
from lenses import lens
import pytest

velocity_refiner = VelocityRefiner()

def create_ball_only_game(ts: float, x: float, y: float, z: float, vx: float = 1, vy: float = 1, vz: float = 1):
    return Game(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={},
        enemy_robots={},
        ball=Ball(vector.obj(x=x, y=y, z=z), vector.obj(x=vx, y=vy, z=vz), None)    
    )

def create_one_robot_only_game(ts: float, x: float, y: float, is_friendly: bool):
    robots = {1: Robot(
        id=1,
        is_friendly=is_friendly,
        has_ball=True,
        p=vector.obj(x=x, y=y),
        v=None,
        a=None,
        orientation=0
    )}
    return Game(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={} if not is_friendly else robots,
        enemy_robots={} if is_friendly else robots,
        ball=None    
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

    target_robot = (game.enemy_robots[1] if not is_friendly else game.friendly_robots[1])

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
    time_velocity_pairs = []
    for i in range(20):
        time = i
        vx, vy, vz = 1, 1, 1
        past_game.add_game(create_ball_only_game(i, i, i, i, vx, vy, vz))
        time_velocity_pairs.append((time, vector.obj(x=vx, y=vy, z=vz)))
    time_velocity_pairs = time_velocity_pairs[::-1]
    game = create_ball_only_game(i + 1, i + 1, i + 1, i + 1)
    extracted = velocity_refiner._extract_time_velocity_pairs(past_game, game, lens.ball)
    assert extracted == time_velocity_pairs[:VelocityRefiner.ACCELERATION_N_WINDOWS * VelocityRefiner.ACCELERATION_WINDOW_SIZE]

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
            past_game.add_game(create_ball_only_game(ts, 0, 0, 0, acc * i + noise, acc * i + noise, acc * i + noise))
            ts += 1

    game = create_ball_only_game(ts, 0, 0, 0)
    game = velocity_refiner.refine(past_game, game)
    
    assert game.ball.a.x == pytest.approx(acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE)
    assert game.ball.a.y == pytest.approx(acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE)
    assert game.ball.a.z == pytest.approx(acc / VelocityRefiner.ACCELERATION_WINDOW_SIZE)
