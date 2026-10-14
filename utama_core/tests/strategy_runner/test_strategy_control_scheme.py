"""Tests for strategy-defined control scheme configuration in AbstractStrategy and StrategyRunner."""

from __future__ import annotations

import pytest

from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.motion_planning.src.controllers import (
    DWAController,
    FastPathPlanningController,
    PIDController,
    TrajectorySamplingController,
)
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.strategy.kernel_strategy import build_default_kernel_strategy
from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy


def _dummy_factory(motion_controller):
    return build_default_kernel_strategy((1, 2))(motion_controller)


class _CustomSchemeStrategy(AbstractStrategy):
    control_scheme = "dwa"


def test_abstract_strategy_control_scheme_precedence():
    # 1. Default when unset is None
    s_default = AbstractStrategy(build_kernel_strategy=_dummy_factory)
    assert s_default.control_scheme is None

    # 2. Inherited from factory attribute
    factory_with_scheme = lambda mc: _dummy_factory(mc)  # noqa: E731
    factory_with_scheme.control_scheme = "trajsample"
    s_factory = AbstractStrategy(build_kernel_strategy=factory_with_scheme)
    assert s_factory.control_scheme == "trajsample"

    # 3. Inherited from class attribute
    s_class = _CustomSchemeStrategy(build_kernel_strategy=_dummy_factory)
    assert s_class.control_scheme == "dwa"

    # 4. Constructor argument overrides class attribute and factory attribute
    s_override = _CustomSchemeStrategy(build_kernel_strategy=factory_with_scheme, control_scheme="pid")
    assert s_override.control_scheme == "pid"


def test_strategy_runner_resolves_strategy_control_scheme():
    # Friendly strategy requests 'trajsample'
    strat = AbstractStrategy(build_kernel_strategy=_dummy_factory, control_scheme="trajsample")
    runner = StrategyRunner(
        strategy=strat,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
    )
    assert runner.control_scheme == "trajsample"
    assert runner.my.control_scheme == "trajsample"
    assert runner.my.motion_controller == TrajectorySamplingController


def test_strategy_runner_defaults_to_fpp_when_unspecified():
    # Neither StrategyRunner nor DummyStrategy specifies a control_scheme
    runner = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
    )
    assert runner.control_scheme == "fpp"
    assert runner.my.control_scheme == "fpp"
    assert runner.my.motion_controller == FastPathPlanningController


def test_strategy_runner_explicit_override_takes_precedence():
    # Strategy declares 'trajsample', but runner explicitly passes 'dwa'
    strat = AbstractStrategy(build_kernel_strategy=_dummy_factory, control_scheme="trajsample")
    runner = StrategyRunner(
        strategy=strat,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        control_scheme="dwa",
    )
    assert runner.control_scheme == "dwa"
    assert runner.my.control_scheme == "dwa"
    assert runner.my.motion_controller == DWAController


def test_strategy_runner_asymmetric_pvp_control_schemes():
    # Friendly uses 'trajsample', opponent uses 'dwa'
    strat_a = AbstractStrategy(build_kernel_strategy=_dummy_factory, control_scheme="trajsample")
    strat_b = AbstractStrategy(build_kernel_strategy=_dummy_factory, control_scheme="dwa")

    runner = StrategyRunner(
        strategy=strat_a,
        opp_strategy=strat_b,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
    )
    assert runner.control_scheme == "trajsample"
    assert runner.opp_control_scheme == "dwa"
    assert runner.my.motion_controller == TrajectorySamplingController
    assert runner.opp.motion_controller == DWAController


def test_strategy_runner_opp_explicit_override():
    # Friendly uses 'trajsample', opponent strategy declares 'dwa', but opp_control_scheme overrides to 'pid'
    strat_a = AbstractStrategy(build_kernel_strategy=_dummy_factory, control_scheme="trajsample")
    strat_b = AbstractStrategy(build_kernel_strategy=_dummy_factory, control_scheme="dwa")

    runner = StrategyRunner(
        strategy=strat_a,
        opp_strategy=strat_b,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        opp_control_scheme="pid",
    )
    assert runner.control_scheme == "trajsample"
    assert runner.opp_control_scheme == "pid"
    assert runner.my.motion_controller == TrajectorySamplingController
    assert runner.opp.motion_controller == PIDController
