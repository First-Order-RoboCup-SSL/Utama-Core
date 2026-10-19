"""Flat OpenJev engine: the model picks every outfield robot's action itself, no tactics in between.

The tactic that asks and carries it out and its menu are shared code (`tactics/flat.py`,
`flat_state.py`); this module builds it with an OpenJev client (FOR-Engine), or with `policy` the
scripted rule in its place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.openjev.client import OpenJevClient
from utama_core.tactics.flat import FlatJevTactic, flat_strategy


def build_openjev_flat_kernel_strategy(
    outfield_robot_ids: tuple[int, ...],
    *,
    url: str = "http://127.0.0.1:3000",
    decision_hz: float = 1.0,
    log_path: Optional[str | Path] = None,
    client: Optional[OpenJevClient] = None,
    policy: str = "openjev",
    call: bool = True,
    on_build: Optional[Callable[[FlatJevTactic], None]] = None,
):
    """OpenJev controls every outfield robot (`tactics/flat.py`). Same shape as the
    `build_*_kernel_strategy` factories; `policy="scripted"` needs no server, `call=False` leaves the
    coaching's call out of the questions."""

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        jev_client = client or (OpenJevClient(url) if policy == "openjev" else None)
        tactic = FlatJevTactic(
            jev_client,
            decision_hz=decision_hz,
            log_path=str(log_path) if log_path else None,
            policy=policy,
            call=call,
        )
        if on_build is not None:
            on_build(tactic)
        return flat_strategy(outfield_robot_ids, tactic, motion_controller)

    return _build
