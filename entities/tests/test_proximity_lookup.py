import numpy as np
import pytest
from entities.game.proximity_lookup import ProximityLookup
from entities.game.robot import Robot
from entities.game.ball import Ball
from entities.data.vector import Vector2D, Vector3D


def test_proximity_lookup_handles_none_inputs():
    lookup = ProximityLookup(friendly_robots=None, enemy_robots=None, ball=None)
    obj, dist = lookup.closest_to_ball()
    assert obj is None
    assert np.isinf(dist)


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
