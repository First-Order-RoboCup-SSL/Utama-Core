"""The strategies: a module in this package with one `build(outfield_robot_ids, motion_controller)`.

The module's name is the strategy's name, so a new strategy is a new file and nothing else.
"""

from __future__ import annotations

from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Callable

_PACKAGE = Path(__file__).parent


def names() -> list[str]:
    """Every strategy, found from source: importing them all would run every module's import-time code."""
    return sorted(p.stem for p in _PACKAGE.glob("*.py") if "\ndef build(" in p.read_text())


TEST_ONLY = "default"  # everyone attacks with one tactic: the tests' minimal strategy, never a player


def player_names() -> list[str]:
    """Every strategy that plays matches: `names()` without the tests' minimal one."""
    return [name for name in names() if name != TEST_ONLY]


# The names strategies had until 2026-11-12, so old replays and summaries still load. Only `builder`
# and `current_name` read this: replay files keep the names they were saved under, so `plain_name`
# (which tools use to find those files) does not.
RENAMED = {
    "split_shape": "pendulum",
    "clear_press_plus": "breakwater",
    "overload_press": "avalanche",
    "high_press": "wolfpack",
    "decoy_and_overload": "mirage",
    "counter_flow": "riptide",
    "score_aware_counter_flow": "chameleon",
    "press_trigger_flow": "tripwire",
    "clear_danger": "firebreak",
    "press_and_pass": "terrier",
    "tiki_taka_plus": "weaver",
    "give_and_go_solo": "ricochet",
}


def current_name(name: str) -> str:
    """`name` as the catalog calls it today: old `build_*_kernel_strategy` and pre-2026-11-12 names map on."""
    plain = plain_name(name)
    return RENAMED.get(plain, plain)


def builder(name: str, outfield_robot_ids: tuple[int, ...]) -> Callable:
    """`AbstractStrategy`'s first argument for strategy `name` (an old name works too)."""
    return partial(import_module(f"utama_core.strategy.{current_name(name)}").build, outfield_robot_ids)


def plain_name(name: str) -> str:
    """`name`, also accepting the `build_<name>_kernel_strategy` of replays recorded before 2026-11-10."""
    return name.removeprefix("build_").removesuffix("_kernel_strategy")
