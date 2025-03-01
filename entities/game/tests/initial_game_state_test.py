from entities.game.game import Game
from entities.game.robot import Robot
from entities.game.ball import Ball
from entities.data.vision import VisionBallData, VisionData, VisionRobotData


blue_robot_data = [VisionRobotData(i, i, 10 * i, 0) for i in range(6)]
yellow_robot_data = [VisionRobotData(i, 100 * i, 1000 * i, 0) for i in range(6)]
basic_ball = VisionBallData(0, 0, 0, 0)


def test_passing_yellow_game_initial_frame_is_reflected_in_robots():
    game = Game(
        True,
        True,
        start_frame=VisionData(0, yellow_robot_data, blue_robot_data, [basic_ball]),
    )

    assert len(game.friendly_robots) == 6
    assert len(game.enemy_robots) == 6

    for i in range(6):
        assert isinstance(game.friendly_robots[i], Robot)
        assert isinstance(game.enemy_robots[i], Robot)

        assert game.friendly_robots[i].x == 100 * i, (
            f"Expected {100 * i}, got {game.friendly_robots[i].x}"
        )
        assert game.friendly_robots[i].y == 1000 * i, (
            f"Expected {1000 * i}, got {game.friendly_robots[i].y}"
        )
        assert game.friendly_robots[i].orientation == 0

        assert game.enemy_robots[i].x == i, (
            f"Expected {i}, got {game.enemy_robots[i].x}"
        )
        assert game.enemy_robots[i].y == 10 * i, (
            f"Expected {10 * i}, got {game.enemy_robots[i].y}"
        )
        assert game.enemy_robots[i].orientation == 0


def test_passing_blue_game_initial_frame_is_reflected_in_robots():
    game = Game(
        False,
        True,
        start_frame=VisionData(0, yellow_robot_data, blue_robot_data, [basic_ball]),
    )

    assert len(game.friendly_robots) == 6
    assert len(game.enemy_robots) == 6

    for i in range(6):
        assert isinstance(game.friendly_robots[i], Robot)
        assert isinstance(game.enemy_robots[i], Robot)

        assert game.friendly_robots[i].x == i, (
            f"Expected {i}, got {game.friendly_robots[i].x}"
        )
        assert game.friendly_robots[i].y == 10 * i, (
            f"Expected {10 * i}, got {game.friendly_robots[i].y}"
        )
        assert game.friendly_robots[i].orientation == 0

        assert game.enemy_robots[i].x == 100 * i, (
            f"Expected {i * 100}, got {game.enemy_robots[i].x}"
        )
        assert game.enemy_robots[i].y == 1000 * i, (
            f"Expected {1000 * i}, got {game.enemy_robots[i].y}"
        )
        assert game.enemy_robots[i].orientation == 0


def test_passing_initial_large_robot_id_is_reflected():
    game = Game(
        True,
        True,
        start_frame=VisionData(
            0, [VisionRobotData(12345, 1, 2, 3)], blue_robot_data, [basic_ball]
        ),
    )

    assert len(game.friendly_robots) == 1
    assert game.friendly_robots[12345].x == 1
    assert game.friendly_robots[12345].y == 2
    assert game.friendly_robots[12345].orientation == 3


def test_passing_initial_ball_is_reflected():
    game = Game(
        True,
        True,
        start_frame=VisionData(
            0, yellow_robot_data, blue_robot_data, [VisionBallData(1, 2, 3, 4)]
        ),
    )

    assert isinstance(game.ball, Ball)
    assert game.ball.x == 1
    assert game.ball.y == 2
    assert game.ball.z == 3


def test_initial_highest_confidence_ball_is_taken():
    game = Game(
        True,
        True,
        start_frame=VisionData(
            0,
            yellow_robot_data,
            blue_robot_data,
            [VisionBallData(1, 2, 3, 4), VisionBallData(10, 20, 30, 40)],
        ),
    )

    assert game.ball.x == 10
    assert game.ball.y == 20
    assert game.ball.z == 30


if __name__ == "__main__":
    test_passing_yellow_game_initial_frame_is_reflected_in_robots()
    test_passing_blue_game_initial_frame_is_reflected_in_robots
    test_passing_initial_large_robot_id_is_reflected()
    test_passing_initial_ball_is_reflected()
    test_initial_highest_confidence_ball_is_taken()
    print("All tests passed!")
