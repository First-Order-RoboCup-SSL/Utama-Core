from entities.game.game import Game
from entities.game.robot import Robot
from entities.game.ball import Ball
from entities.data.vision import BallData, FrameData, RobotData


blue_robot_data = [RobotData(i, i, 10*i, 0) for i in range(6)]
yellow_robot_data = [RobotData(i, 100*i, 1000*i, 0) for i in range(6)]
basic_ball = BallData(0, 0, 0, 0)




def test_update_single_friendly_robot():
    game = Game(True, True, start_frame=FrameData(1, yellow_robot_data, blue_robot_data, [basic_ball]))

    assert game.friendly_robots[1].x == 100
    assert game.friendly_robots[1].y == 1000
    assert game.friendly_robots[1].orientation == 0

    game.add_new_state(FrameData(0, [RobotData(1, -1, -2, -3)], [], []))

    assert game.friendly_robots[1].x == -1
    assert game.friendly_robots[1].y == -2
    assert game.friendly_robots[1].orientation == -3

def test_update_single_large_friendly_robot():
    game = Game(True, True, start_frame=FrameData(1, [RobotData(12345, 1,2,3)], blue_robot_data, [basic_ball]))

    assert game.friendly_robots[12345].x == 1
    assert game.friendly_robots[12345].y == 2
    assert game.friendly_robots[12345].orientation == 3

    game.add_new_state(FrameData(0, [RobotData(12345, -1, -2, -3)], [], []))

    assert game.friendly_robots[12345].x == -1
    assert game.friendly_robots[12345].y == -2
    assert game.friendly_robots[12345].orientation == -3

def test_update_single_enemy_robot():
    game = Game(True, True, start_frame=FrameData(0, yellow_robot_data, blue_robot_data, [basic_ball]))
    game.add_new_state(FrameData(0, [], [RobotData(0, -1, -2, -3)], []))

    assert game.enemy_robots[0].id == 0
    assert game.enemy_robots[0].x == -1
    assert game.enemy_robots[0].y == -2
    assert game.enemy_robots[0].orientation == -3

def test_update_empty_frame():
    game = Game(True, True, start_frame=FrameData(0, yellow_robot_data, blue_robot_data, [basic_ball]))
    previous_friendly = game.friendly_robots
    previous_enemy = game.enemy_robots
    previous_ball = game.ball
    game.add_new_state(FrameData(0, [], [], []))

    assert previous_friendly == game.friendly_robots
    assert previous_enemy == game.enemy_robots
    assert previous_ball == game.ball