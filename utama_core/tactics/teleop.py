"""Driving a robot by hand from the dashboard (the Live tab's Teleop panel, `examples/teleop.py`).

The page sends what its keys ask for to `TeleopDrive.set` several times a second while a key is
held; each robot's `TeleopTactic` plays the latest, in the robot's own frame. A robot stops when
the messages stop for `DEADMAN_S` (the page closed, the network dropped), and, like any tactic,
whenever the referee halts or stops play. A kick fires once per press.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game

DEADMAN_S = 0.3
MAX_SPEED = 1.0  # m/s, each axis
MAX_TURN = 3.0  # rad/s

_STILL = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)


def _clamp(value, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


class TeleopDrive:
    """The latest command the page sent for each robot (named as the page lists them, "yellow 0")."""

    def __init__(self, robots: list[str], clock: Callable[[], float] = time.monotonic):
        self.robots = list(robots)
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[float, RobotCommand]] = {}

    def set(self, robot: str, forward=0.0, left=0.0, turn=0.0, kick=False, dribble=False) -> None:
        if robot not in self.robots:
            raise ValueError(f"no robot {robot!r} here")
        command = RobotCommand(
            _clamp(forward, MAX_SPEED),
            _clamp(left, MAX_SPEED),
            _clamp(turn, MAX_TURN),
            kick=bool(kick),
            chip=False,
            dribble=bool(dribble),
        )
        with self._lock:
            # A kick asked for and not yet played stays asked for until played.
            pending_kick = robot in self._latest and self._latest[robot][1].kick
            self._latest[robot] = (self._clock(), command._replace(kick=command.kick or pending_kick))

    def command(self, robot: str) -> RobotCommand:
        """What `robot` plays this tick: still once the page has gone quiet; a kick only once."""
        with self._lock:
            latest = self._latest.get(robot)
            if latest is None or self._clock() - latest[0] > DEADMAN_S:
                return _STILL
            stamp, command = latest
            if command.kick:
                self._latest[robot] = (stamp, command._replace(kick=False))
            return command


class TeleopTactic(BaseTactic[None]):
    """Plays `drive`'s command for `robot` on its first robot; any others stand still."""

    tag = TacticTag.MIXED

    def __init__(self, drive: TeleopDrive, robot: str):
        self.drive = drive
        self.robot = robot

    def initial_mem(self) -> None:
        return None

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        commands = {rid: _STILL for rid in robot_ids}
        if robot_ids:
            commands[robot_ids[0]] = self.drive.command(self.robot)
        return commands, mem
