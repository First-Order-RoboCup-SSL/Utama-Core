import sys
import types

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.skills.src.score_goal import _find_best_shot

# Stub out heavy simulator dependency before importing the module under test
module = types.ModuleType("rsoccer_simulator.src.ssl.envs.standard_ssl")
module.SSLStandardEnv = object
sys.modules.setdefault("rsoccer_simulator", types.ModuleType("rsoccer_simulator"))
sys.modules.setdefault("rsoccer_simulator.src", types.ModuleType("rsoccer_simulator.src"))
sys.modules.setdefault("rsoccer_simulator.src.ssl", types.ModuleType("rsoccer_simulator.src.ssl"))
sys.modules.setdefault("rsoccer_simulator.src.ssl.envs", types.ModuleType("rsoccer_simulator.src.ssl.envs"))
sys.modules["rsoccer_simulator.src.ssl.envs.standard_ssl"] = module

import utama_core.skills.src.score_goal as sg  # noqa: E402


def test_candidate_stays_within_boundary_gap(monkeypatch):
    """When the largest gap touches a goal boundary, the chosen shot should
    be offset slightly inside the interval and remain within the gap."""

    def fake_ray_casting(point, enemy_robots, goal_x, goal_y1, goal_y2):
        return [(-0.2, 0.2)]  # central shadow

    monkeypatch.setattr(sg, "_ray_casting", fake_ray_casting)

    point = Vector2D(0, 0)
    best, gap = sg._find_best_shot(point, [], goal_x=1.0, goal_y1=-1.0, goal_y2=1.0)

    assert gap == (-1.0, -0.2)
    assert gap[0] < best < gap[1]


def test_prefers_largest_open_gap(monkeypatch):
    """The algorithm should select a candidate within the largest open
    interval on the goal line."""

    def fake_ray_casting(point, enemy_robots, goal_x, goal_y1, goal_y2):
        # Shadows leave open spaces: (-1,-0.5), (-0.1,0.2), (0.3,1)
        return [(-0.5, -0.1), (0.2, 0.3)]

    monkeypatch.setattr(sg, "_ray_casting", fake_ray_casting)

    point = Vector2D(0, 0)
    best, gap = sg._find_best_shot(point, [], goal_x=1.0, goal_y1=-1.0, goal_y2=1.0)

    assert gap == (0.3, 1.0)
    assert gap[0] < best < gap[1]
    assert np.isclose(best, 0.86, rtol=1e-2)
