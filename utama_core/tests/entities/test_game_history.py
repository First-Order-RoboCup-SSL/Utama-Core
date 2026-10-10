import numpy as np

from utama_core.entities.data.object import ObjectKey, ObjectType, TeamType
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, GameHistory, Robot
from utama_core.entities.game.game_history import AttributeType

ZERO = Vector2D(0, 0)


def _robot(rid, is_friendly, x, v=ZERO) -> Robot:
    return Robot(rid, is_friendly, False, Vector2D(x, 1.0), v, ZERO, 0.0)


def test_stores_position_and_velocity_per_object_as_float_tuples():
    ball = Ball(Vector3D(0.1, 0.2, 0.0), Vector3D(1.0, 0.0, 0.0), Vector3D(0, 0, 0))
    frame = GameFrame(
        0.5, True, True, {0: _robot(0, True, 2.0, Vector2D(0.5, -0.5))}, {3: _robot(3, False, -1.0)}, ball
    )
    history = GameHistory(4)
    history.add_game_frame(frame)

    friendly = ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, 0)
    assert list(history.historical_data) == [
        ObjectKey(TeamType.NEUTRAL, ObjectType.BALL, 0),
        friendly,
        ObjectKey(TeamType.ENEMY, ObjectType.ROBOT, 3),
    ]
    assert list(history.historical_data[friendly]) == [AttributeType.POSITION, AttributeType.VELOCITY]
    assert history.get_historical_attribute_entries(friendly, AttributeType.VELOCITY, 1) == [(0.5, (0.5, -0.5))]
    assert history.get_historical_attribute_entries(
        ObjectKey(TeamType.NEUTRAL, ObjectType.BALL, 0), AttributeType.POSITION, 1
    ) == [(0.5, (0.1, 0.2, 0.0))]
    assert history.raw_games_history[-1] is frame


def test_a_missing_vector_stores_nothing_and_a_non_int_id_is_skipped():
    no_velocity = Robot(1, True, False, Vector2D(1.0, 1.0), None, ZERO, 0.0)
    numpy_id = _robot(np.int64(2), False, 0.0)
    nothing = Robot(4, True, False, None, None, ZERO, 0.0)
    history = GameHistory(4)
    history.add_game_frame(GameFrame(0.0, True, True, {1: no_velocity, 4: nothing}, {2: numpy_id}, None))

    key = ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, 1)
    assert list(history.historical_data) == [key]  # no entry for robot 4, none for the numpy id
    assert list(history.historical_data[key]) == [AttributeType.POSITION]
