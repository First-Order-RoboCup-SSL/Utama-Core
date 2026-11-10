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


def builder(name: str, outfield_robot_ids: tuple[int, ...]) -> Callable:
    """`AbstractStrategy`'s first argument for strategy `name`."""
    return partial(import_module(f"utama_core.strategy.{plain_name(name)}").build, outfield_robot_ids)


def plain_name(name: str) -> str:
    """`name`, also accepting the `build_<name>_kernel_strategy` of replays recorded before 2026-11-10."""
    return name.removeprefix("build_").removesuffix("_kernel_strategy")
