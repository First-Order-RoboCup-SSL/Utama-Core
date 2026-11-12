"""The one way to load a strategy and to set up a simulated match. Every tool that plays one uses these:
`tools/evaluation/match.py` (round-robin, `evaluate.py`), `repro_from_replay.py`, `moments.py`,
`debug_match.py` and `main.py`.

- `load_side(name)`: a strategy by catalog name, a `.strategy` file by path, or `PATH.py:FUNCTION` (a
  `build(outfield_robot_ids, motion_controller)` in a scratch file, to try a tactic without a repo change).
- `sim_referee(...)`: the referee of an rsim match: the "simulation" profile, with the kickoff team.
- `sim_runner(strategy, opponent, ...)`: the `StrategyRunner` for a 6-a-side rsim match against a referee
  (`sim_referee()` by default), our side yellow on the right, vision stream off.

Nothing else constructs the referee or the rsim runner of a match. Changing this module changes the
fingerprint of every match (`utama_core/replay/fingerprint.py`), so a round-robin plays them all again.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from utama_core.bundle.strategy_file import SUFFIX, BundledStrategy
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.rsoccer_simulator.Utils.sim_noise import SimNoise
from utama_core.run import StrategyRunner
from utama_core.strategy import catalog

N_OUTFIELD = 5  # + 1 goalkeeper per side
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))
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
        if catalog.current_name(name) not in catalog.names():
            raise UnknownStrategy(f"unknown strategy {name!r}; strategies: {', '.join(catalog.names())}")
        return partial(catalog.builder, catalog.current_name(name))
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


def load_side(name: str, outfield_robot_ids: Optional[tuple[int, ...]] = None) -> AbstractStrategy:
    """A strategy to play: a `.strategy` file by path, `PATH.py:FUNCTION`, or a catalog name. The robots
    are the outfield ones unless given; a `.strategy` file plays the robots it was built for unless given."""
    if name.endswith(SUFFIX):
        if not Path(name).is_file():
            raise UnknownStrategy(f"{name}: no such strategy file")
        return BundledStrategy(name, outfield_robot_ids)
    return AbstractStrategy(build=factory_for(name)(outfield_robot_ids or OUTFIELD_ROBOT_IDS))


def sim_referee(*, kickoff_team: str = "yellow") -> CustomReferee:
    """The referee of an rsim match: the "simulation" profile, `kickoff_team` taking the first kickoff."""
    profile = load_profile("simulation")
    if kickoff_team != profile.game.kickoff_team:
        profile = dataclasses.replace(profile, game=dataclasses.replace(profile.game, kickoff_team=kickoff_team))
    return CustomReferee(profile, n_robots_yellow=N_ROBOTS, n_robots_blue=N_ROBOTS)


def sim_runner(
    strategy: AbstractStrategy,
    opponent: AbstractStrategy,
    *,
    sim_noise: Optional[SimNoise] = None,
    initial_command: Optional[RefereeCommand] = RefereeCommand.PREPARE_KICKOFF_YELLOW,
    referee: Optional[CustomReferee] = None,
    my_team_is_right: bool = True,
    **runner_kwargs,
) -> tuple[StrategyRunner, CustomReferee]:
    """A `StrategyRunner` for `strategy` (yellow) against `opponent` in rsim, refereed by `referee` (default
    `sim_referee()`). `initial_command=None` keeps the runner's own default. Other keyword arguments go to
    `StrategyRunner` (`match_log_path`, `stats_path`, `control_scheme`, `show_live_status`, ...)."""
    if referee is None:
        referee = sim_referee()
    if initial_command is not None:
        runner_kwargs["referee_initial_command"] = initial_command
    runner = StrategyRunner(
        strategy=strategy,
        opp_strategy=opponent,
        my_team_is_yellow=True,
        my_team_is_right=my_team_is_right,
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
