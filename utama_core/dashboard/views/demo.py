"""Demo view — switch which demo a long-running process plays, from the Live tab.

`examples/demos.py` keeps one dashboard up and plays one demo (a runner built by an example's
`build_runner`) at a time. The page lists the demos with a Run button; a request, only while
HALTed, stops the running demo (`StrategyRunner.stop`: robots stopped, log closed, radio and
vision released) and the script builds the chosen one on the same dashboard. Each demo started
bumps `generation`; the page polls `/demo/config` and reloads itself when it changes, so its
field, referee and strategy panels are the new demo's with no manual refresh.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Optional

from utama_core.dashboard.server import DashboardServer

if TYPE_CHECKING:
    from utama_core.run import StrategyRunner


class DemoSwitcher:
    def __init__(self, names: list[str]):
        self.names = list(names)
        self.running: Optional[str] = None
        self.runner: Optional["StrategyRunner"] = None
        self.wanted: Optional[str] = None  # asked for, not started yet
        self.error: Optional[str] = None  # why the last demo asked for did not start
        self.generation = 0
        self._lock = threading.Lock()
        self._asked = threading.Event()

    def config(self) -> dict:
        with self._lock:
            return {
                "choices": self.names,
                "running": self.running,
                "switching": self.wanted,
                "error": self.error,
                "generation": self.generation,
            }

    def request(self, name: str) -> None:
        """Ask for demo `name` (from the page); refused (ValueError) unless the running one is halted."""
        if name not in self.names:
            raise ValueError(f"no demo {name!r}")
        with self._lock:
            if self.runner is not None and not self.runner.halted:
                raise ValueError("press Halt first: a demo is only switched while halted")
            self.wanted, self.error = name, None
            runner = self.runner
        if runner is not None:
            runner.stop()
        self._asked.set()

    def started(self, name: str, runner: "StrategyRunner") -> None:
        with self._lock:
            self.running, self.runner, self.wanted, self.error = name, runner, None, None
            self.generation += 1

    def failed(self, name: str, error: str) -> None:
        with self._lock:
            self.running, self.runner, self.wanted, self.error = None, None, None, f"{name}: {error}"
            self.generation += 1

    def take(self) -> Optional[str]:
        """The demo asked for since the last one started, or None (it was stopped otherwise: Ctrl+C)."""
        with self._lock:
            self._asked.clear()
            return self.wanted

    def wait(self, poll_s: float = 0.5) -> str:
        """Blocks until the page asks for a demo (Ctrl+C still ends it); that demo's name."""
        while not self._asked.wait(poll_s):
            pass
        return self.take()


def attach(server: DashboardServer, switcher: DemoSwitcher) -> DemoSwitcher:
    server.add_route("/demo/config", lambda: json.dumps(switcher.config()).encode())
    server.add_command_handler("/demo/run", lambda payload: switcher.request(str(payload["name"])))
    return switcher
