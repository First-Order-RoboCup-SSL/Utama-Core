import numpy as np
import pytest

from utama_core.entities.data.vector import Vector2D
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
