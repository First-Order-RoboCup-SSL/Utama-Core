"""`StrategyRunner` gives a team's motion controller its real robots' kick bends, in rad per
metre off-centre (`RobotProfile.kick_bend_deg_per_cm`), and biases, in rad
(`kick_bias_deg`), and none in the simulators; likewise its vision the tags' offsets."""

import math
from types import SimpleNamespace

import pytest

from utama_core.config.enums import Mode
from utama_core.motion_planning.src.controllers.fastpathplanning import (
    FastPathPlanningController,
)
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.team_controller.src.controllers.common.robot_profile import RobotProfile

_PROFILES = {1: RobotProfile(kick_bend_deg_per_cm=4.7), 2: RobotProfile(), 3: RobotProfile(kick_bias_deg=2.3)}


@pytest.mark.parametrize("mode, expected", [(Mode.REAL, {1: math.radians(4.7) * 100.0}), (Mode.RSIM, {})])
def test_real_robots_get_their_kick_bend_and_the_sims_none(mode, expected):
    runner = SimpleNamespace(mode=mode, rsim_env=None)
    team = SimpleNamespace(motion_controller=FastPathPlanningController)

    controller = StrategyRunner._motion_controller(runner, team, _PROFILES)

    assert controller.kick_bend == pytest.approx(expected)


@pytest.mark.parametrize("mode, expected", [(Mode.REAL, {3: math.radians(2.3)}), (Mode.RSIM, {})])
def test_real_robots_get_their_kick_bias_and_the_sims_none(mode, expected):
    runner = SimpleNamespace(mode=mode, rsim_env=None)
    team = SimpleNamespace(motion_controller=FastPathPlanningController)

    controller = StrategyRunner._motion_controller(runner, team, _PROFILES)

    assert controller.kick_bias == pytest.approx(expected)


def test_real_robots_get_their_tag_offsets_and_the_sims_none():
    tagged = {0: RobotProfile(tag_left_m=-0.002), 1: RobotProfile(kick_bend_deg_per_cm=4.7)}
    real = SimpleNamespace(mode=Mode.REAL, yellow_robot_profiles=tagged, blue_robot_profiles=tagged)

    assert StrategyRunner._tag_offsets(real) == pytest.approx({(True, 0): (0.0, -0.002), (False, 0): (0.0, -0.002)})
    assert StrategyRunner._tag_offsets(SimpleNamespace(mode=Mode.RSIM)) == {}


def test_real_robots_get_their_own_carry_caps_and_the_team_cap_where_unset():
    runner = SimpleNamespace(mode=Mode.REAL, rsim_env=None)
    team = SimpleNamespace(motion_controller=FastPathPlanningController)
    profiles = {0: RobotProfile(carry_speed=0.7), 1: RobotProfile(carry_turn=3.0), 2: RobotProfile()}

    controller = StrategyRunner._motion_controller(runner, team, profiles)

    assert controller.robot_carry == {0: (0.7, controller.carry_turn), 1: (controller.carry_speed, 3.0)}
    assert (
        StrategyRunner._motion_controller(SimpleNamespace(mode=Mode.RSIM, rsim_env=None), team, profiles).robot_carry
        == {}
    )
