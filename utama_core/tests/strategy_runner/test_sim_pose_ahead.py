"""rsim with simulated delay compensates for it the way real robots do; without delay it must not."""

from types import SimpleNamespace

import pytest

from utama_core.config.enums import Mode
from utama_core.config.robot_params import REAL_PARAMS
from utama_core.motion_planning.controllers.fastpathplanning import (
    FastPathPlanningController,
)
from utama_core.rsoccer_simulator.Utils.sim_noise import SimNoise
from utama_core.run.strategy_runner import StrategyRunner

TEAM = SimpleNamespace(motion_controller=FastPathPlanningController)


def latency(mode, rsim_env):
    runner = SimpleNamespace(mode=mode, rsim_env=rsim_env)
    return StrategyRunner._motion_controller(runner, TEAM, {}).pose_ahead.latency_s


def test_rsim_looks_ahead_by_the_middle_of_its_simulated_delay():
    env = SimpleNamespace(sim_noise=SimNoise(latency_s=(0.05, 0.09)))
    assert latency(Mode.RSIM, env) == pytest.approx(0.07)


@pytest.mark.parametrize("env", [None, SimpleNamespace(sim_noise=None), SimpleNamespace(sim_noise=SimNoise())])
def test_rsim_without_delay_does_not_look_ahead(env):
    assert latency(Mode.RSIM, env) == 0.0


def test_real_robots_keep_their_own_setting():
    assert latency(Mode.REAL, None) == REAL_PARAMS.CONTROL_LATENCY_S
