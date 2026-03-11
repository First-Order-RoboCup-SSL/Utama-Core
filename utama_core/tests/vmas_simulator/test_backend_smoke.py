"""Smoke tests for the VMAS SSL backend."""

from utama_core.rsoccer_simulator.src.Entities import Robot
from utama_core.vmas_simulator.src.Simulators.vmas_ssl import VmasSSL
from utama_core.vmas_simulator.src.Utils.config import SSLScenarioConfig


def test_vmas_ssl_backend_smoke():
    sim = VmasSSL(
        n_robots_blue=1,
        n_robots_yellow=1,
        num_envs=1,
        device="cpu",
        scenario_config=SSLScenarioConfig(n_blue=1, n_yellow=1),
    )
    try:
        initial = sim.get_frame()
        assert len(initial.robots_blue) == 1
        assert len(initial.robots_yellow) == 1

        sim.send_commands(
            [
                Robot(yellow=False, id=0, v_x=1.0, v_y=0.0, v_theta=0.0),
                Robot(yellow=True, id=0, v_x=0.0, v_y=0.0, v_theta=0.0),
            ]
        )
        updated = sim.get_frame()
        assert updated.robots_blue[0].v_x >= 0.0
        assert updated.ball is not None
    finally:
        sim.stop()
