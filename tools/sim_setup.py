"""One way to load a strategy and set up a simulated match, for the tools that play from a position or
for a while (`repro_from_replay.py`, `moments.py`, `debug_match.py`, `main.py`).

- `load_side(name)`: a strategy by catalog name, a `.strategy` file by path, or `PATH.py:FUNCTION` (a
  `build(outfield_robot_ids, motion_controller)` in a scratch file, to try a tactic without a repo change).
- `sim_runner(strategy, opponent, ...)`: the referee and the `StrategyRunner` for a 6-a-side rsim match,
  our side yellow on the right, vision stream off.

`tools/evaluation/match.py` still builds its own runner: it is part of every cached match's fingerprint
(`utama_core/replay/fingerprint.py`), so changing it replays every cached match and starts the
evaluation ledger afresh. Move it onto these helpers together with the next change that forces that anyway.
"""

from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from tools.evaluation.match import N_OUTFIELD, OUTFIELD_ROBOT_IDS
from utama_core.bundle.strategy_file import SUFFIX, BundledStrategy
from utama_core.custom_referee import CustomReferee
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise
from utama_core.run import StrategyRunner
from utama_core.strategy import catalog

N_ROBOTS = N_OUTFIELD + 1  # per side, the goalkeeper included


class UnknownStrategy(ValueError):
    """A strategy name that is neither in the catalog, a `.strategy` file nor `PATH.py:FUNCTION`."""


def is_file_spec(name: str) -> bool:
    """`PATH.py:FUNCTION`, a factory in a scratch file."""
    path, sep, func = name.rpartition(":")
    return bool(sep) and path.endswith(".py") and bool(func)


def factory_for(name: str) -> Callable:
    """`f(outfield_robot_ids)` -> `AbstractStrategy`'s `build`, for a catalog strategy or `PATH.py:FUNCTION`
    (imported as it is)."""
    if not is_file_spec(name):
        if catalog.plain_name(name) not in catalog.names():
            raise UnknownStrategy(f"unknown strategy {name!r}; strategies: {', '.join(catalog.names())}")
        return partial(catalog.builder, catalog.plain_name(name))
    path, _, func = name.rpartition(":")
    if not Path(path).is_file():
        raise UnknownStrategy(f"{name}: no such file {path}")
    spec = importlib.util.spec_from_file_location(f"scratch_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, func, None)
    if not callable(factory):
        raise UnknownStrategy(f"{name}: {path} has no factory {func!r}")
    return lambda outfield_robot_ids: partial(factory, outfield_robot_ids)


def load_side(name: str, outfield_robot_ids: tuple[int, ...] = OUTFIELD_ROBOT_IDS) -> AbstractStrategy:
    """A strategy to play: a `.strategy` file by path, `PATH.py:FUNCTION`, or a catalog name."""
    if name.endswith(SUFFIX):
        if not Path(name).is_file():
            raise UnknownStrategy(f"{name}: no such strategy file")
        return BundledStrategy(name, outfield_robot_ids)
    return AbstractStrategy(build=factory_for(name)(outfield_robot_ids))


def sim_runner(
    strategy: AbstractStrategy,
    opponent: AbstractStrategy,
    *,
    sim_noise: Optional[SimNoise] = None,
    initial_command: Optional[RefereeCommand] = RefereeCommand.PREPARE_KICKOFF_YELLOW,
    **runner_kwargs,
) -> tuple[StrategyRunner, CustomReferee]:
    """The referee ("simulation" profile) and a `StrategyRunner` for `strategy` (yellow, on the right)
    against `opponent` in rsim. `initial_command=None` keeps the runner's own default. Other keyword
    arguments go to `StrategyRunner` (`match_log_path`, `stats_path`, `control_scheme`, ...)."""
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=N_ROBOTS, n_robots_blue=N_ROBOTS)
    if initial_command is not None:
        runner_kwargs["referee_initial_command"] = initial_command
    runner = StrategyRunner(
        strategy=strategy,
        opp_strategy=opponent,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_ROBOTS,
        exp_enemy=N_ROBOTS,
        exp_ball=True,
        referee=referee,
        enable_vision_stream=False,
        sim_noise=sim_noise,
        **runner_kwargs,
    )
    return runner, referee
