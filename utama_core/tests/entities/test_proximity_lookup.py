import numpy as np
import pytest

from utama_core.entities.data.object import ObjectKey, ObjectType, TeamType
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.proximity_lookup import ProximityLookup
from utama_core.entities.game.robot import Robot


@pytest.mark.filterwarnings("ignore:Invalid closest_to_ball query")  # Ignore warning about no ball
def test_proximity_lookup_handles_none_inputs():
    lookup = ProximityLookup(friendly_robots=None, enemy_robots=None, ball=None)
    obj, dist = lookup.closest_to_ball()
    assert obj is None
    assert np.isinf(dist)


@pytest.mark.filterwarnings("ignore:Invalid closest_to_ball query")  # Ignore warning about no ball
def test_proximity_lookup_with_no_ball():
    robots = {
        1: Robot(
            id=1,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(0, 0),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0,
        )
    }
    lookup = ProximityLookup(friendly_robots=robots, enemy_robots=None, ball=None)
    obj, dist = lookup.closest_to_ball()
    assert obj is None
    assert np.isinf(dist)


def _robots(is_friendly: bool, xs: list[float]) -> dict[int, Robot]:
    zero = Vector2D(0, 0)
    return {i: Robot(i, is_friendly, False, Vector2D(x, 0.3 * i), zero, zero, 0) for i, x in enumerate(xs)}


def test_built_on_first_query_and_answers_as_if_built_eagerly():
    friendly, enemy = _robots(True, [0.0, 1.0, -2.0]), _robots(False, [0.5, 3.0])
    ball = Ball(Vector3D(0.4, 0.2, 0), Vector3D(0, 0, 0), Vector3D(0, 0, 0))
    lookup = ProximityLookup(friendly, enemy, ball)
    assert "proximity_matrix" not in vars(lookup)  # most frames are never queried: build nothing yet

    eager = ProximityLookup(friendly, enemy, ball)
    eager._build()
    for team in (None, TeamType.FRIENDLY, TeamType.ENEMY):
        assert lookup.closest_to_ball(team) == eager.closest_to_ball(team)
        assert lookup.closest_to_robot(ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, 1), team) == (
            eager.closest_to_robot(ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, 1), team)
        )
    np.testing.assert_array_equal(lookup.proximity_matrix, eager.proximity_matrix)
