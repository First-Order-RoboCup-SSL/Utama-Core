from game import Game
from robot import Robot
from ball import Ball
from data.vision import BallData, FrameData, RobotData


blue_robot_data = [RobotData(i, i, 10*i, 0) for i in range(6)]
yellow_robot_data = [RobotData(i, 100*i, 1000*i, 0) for i in range(6)]


def test_passing_game_initial_frame_is_reflected_in_robots():
    game = Game(True, True, start_frame=FrameData(0, yellow_robot_data, blue_robot_data, []))

    assert len(game.friendly_robots) == 3
    assert len(game.enemy_robots) == 3

    for i in range(6):
        assert isinstance(game.friendly_robots[i], Robot)
        assert isinstance(game.enemy_robots[i], Robot)

        assert game.friendly_robots[i].x == 100*i
        assert game.friendly_robots[i].y == 1000*i
        assert game.friendly_robots[i].orientation == 0

        assert game.enemy_robots[i].x == i
        assert game.enemy_robots[i].x == 10*i
        assert game.enemy_robots[i].orientation == 0



def test_passing_initial_large_robot_id_is_reflected():
    game = Game(True, True, start_frame=FrameData(0, [RobotData(12345, 1,2,3)], [], []))

    assert len(game.friendly_robots) == 1
    assert game.friendly_robots[12345].x == 1
    assert game.friendly_robots[12345].y == 2
    assert game.friendly_robots[12345].orientation == 3


def test_passing_initial_ball_is_reflected():
    game = Game(True, True, start_frame=FrameData(0, [], [], [BallData(1,2,3,4)]))

    assert isinstance(game.ball, Ball)
    assert game.ball.x == 1
    assert game.ball.y == 2
    assert game.ball.z == 3



def test_initial_highest_confidence_ball_is_taken():
    game = Game(True, True, start_frame=FrameData(0, [], [], [BallData(1,2,3,4), BallData(10, 20, 30, 40)]))

    assert game.ball.x == 10
    assert game.ball.y == 20
    assert game.ball.z == 30

