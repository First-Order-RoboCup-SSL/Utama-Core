"""Flat OpenJev engine: the model picks every outfield robot's action itself, no tactics in between.

The tactic that asks, carries out and records it, its menu and the policy network are shared code
(`tactics/flat.py`, `flat_state.py`, `flat_net.py`); this module builds it with an OpenJev client
(FOR-Engine), or with `policy` the scripted rule or the network in its place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.openjev.client import OpenJevClient
from utama_core.tactics import flat_net
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
    weights: Optional[str | Path] = None,
    record_path: Optional[str | Path] = None,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    on_build: Optional[Callable[[FlatJevTactic], None]] = None,
):
    """OpenJev controls every outfield robot (`tactics/flat.py`). Same shape as the
    `build_*_kernel_strategy` factories; `policy="scripted"` needs no server, `call=False` leaves the
    coaching's call out of the questions, `policy="net"` plays the network in `weights`, and
    `record_path` saves every question the rule or the network answers."""

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        jev_client = client or (OpenJevClient(url) if policy == "openjev" else None)
        tactic = FlatJevTactic(
            jev_client,
            decision_hz=decision_hz,
            log_path=str(log_path) if log_path else None,
            policy=policy,
            call=call,
            net=flat_net.FlatNet.load(weights) if weights else None,
            record_path=str(record_path) if record_path else None,
            temperature=temperature,
            seed=seed,
        )
        if on_build is not None:
            on_build(tactic)
        return flat_strategy(outfield_robot_ids, tactic, motion_controller)

    return _build
