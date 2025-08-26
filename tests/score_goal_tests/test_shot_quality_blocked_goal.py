import sys
import types
from types import SimpleNamespace

from entities.data.vector import Vector2D


def _patch_ssl_env(monkeypatch):
    rsim_mod = types.ModuleType("rsoccer_simulator")
    rsim_src = types.ModuleType("rsoccer_simulator.src")
    rsim_ssl = types.ModuleType("rsoccer_simulator.src.ssl")
    rsim_envs = types.ModuleType("rsoccer_simulator.src.ssl.envs")
    rsim_standard = types.ModuleType(
        "rsoccer_simulator.src.ssl.envs.standard_ssl"
    )

    class DummyEnv:
        pass

    rsim_standard.SSLStandardEnv = DummyEnv
    rsim_envs.standard_ssl = rsim_standard
    rsim_ssl.envs = rsim_envs
    rsim_src.ssl = rsim_ssl
    rsim_mod.src = rsim_src

    monkeypatch.setitem(sys.modules, "rsoccer_simulator", rsim_mod)
    monkeypatch.setitem(sys.modules, "rsoccer_simulator.src", rsim_src)
    monkeypatch.setitem(sys.modules, "rsoccer_simulator.src.ssl", rsim_ssl)
    monkeypatch.setitem(sys.modules, "rsoccer_simulator.src.ssl.envs", rsim_envs)
    monkeypatch.setitem(
        sys.modules,
        "rsoccer_simulator.src.ssl.envs.standard_ssl",
        rsim_standard,
    )


class DummyRobot:
    def __init__(self, x: float, y: float):
        self.p = SimpleNamespace(x=x, y=y)


def test_find_shot_quality_blocked_goal(monkeypatch):
    _patch_ssl_env(monkeypatch)
    from skills.src.score_goal import find_shot_quality

    ball = Vector2D(0, 0)
    goal_x = 10
    goal_y1 = -0.5
    goal_y2 = 0.5
    defenders = [
        DummyRobot(9, y)
        for y in [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45]
    ]

    assert find_shot_quality(ball, defenders, goal_x, goal_y1, goal_y2) == 0.0
