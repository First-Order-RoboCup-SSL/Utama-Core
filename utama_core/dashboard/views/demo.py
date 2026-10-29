"""Demo view — switch which demo a long-running process plays, from the Live tab.

`examples/demos.py` keeps one dashboard up and plays one demo (a runner built by an example's
`build_runner`) at a time. The page lists the demos with a Run button; a request, only while
HALTed, stops the running demo (`StrategyRunner.stop`: robots stopped, log closed, radio and
vision released) and the script builds the chosen one on the same dashboard. Each demo started
bumps `generation`; the page polls `/demo/config` and reloads itself when it changes, so its
field, referee and strategy panels are the new demo's with no manual refresh.

Two more requests ride on the same switch:
- settings (`settings`: name -> the values it takes): sent with Run, they apply from the demo
  it starts (the script sets the examples' config block from them);
- the field check (`checks`: real robots only): stops the demo as a switch does, the script runs
  `tools/field_check.py`'s check with the radio free (each robot turns on the spot), its lines
  shown under the panel, then starts the same demo again.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Optional

from utama_core.dashboard.server import DashboardServer

if TYPE_CHECKING:
    from utama_core.run import StrategyRunner


class DemoSwitcher:
    def __init__(self, names: list[str], settings: Optional[dict[str, list[str]]] = None, checks: bool = False):
        self.names = list(names)
        self.choices = dict(settings or {})  # setting -> its values, the first the default
        self.settings = {name: values[0] for name, values in self.choices.items()}
        self.checks = checks
        self.running: Optional[str] = None
        self.runner: Optional["StrategyRunner"] = None
        self.wanted: Optional[str] = None  # asked for, not started yet
        self.check_wanted = False
        self.check_lines: list[str] = []  # the last field check's, as it runs
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
                "settings": {name: {"values": v, "value": self.settings[name]} for name, v in self.choices.items()},
                "checks": self.checks,
                "checking": self.check_wanted,
                "check_lines": list(self.check_lines),
            }

    def request(self, name: str, settings: Optional[dict] = None) -> None:
        """Ask for demo `name` (from the page), with `settings` from then on; refused (ValueError)
        unless the running one is halted."""
        if name not in self.names:
            raise ValueError(f"no demo {name!r}")
        settings = dict(settings or {})
        for key, value in settings.items():
            if value not in self.choices.get(key, ()):
                raise ValueError(f"no setting {key}={value!r}")
        self._ask(name, settings, check=False)

    def request_check(self) -> None:
        """Ask for the field check, then the running demo again; refused like a switch, and in
        a run whose robots are not real."""
        if not self.checks:
            raise ValueError("the field check is for real robots")
        self._ask(self.running, {}, check=True)

    def _ask(self, name: Optional[str], settings: dict, check: bool) -> None:
        with self._lock:
            if self.runner is not None and not self.runner.halted:
                raise ValueError("press Halt first: a demo is only switched while halted")
            if self.wanted is not None or self.check_wanted:
                raise ValueError("already switching")
            self.settings.update(settings)
            self.wanted, self.error, self.check_wanted = name, None, check
            if check:
                self.check_lines = []
            runner = self.runner
        if runner is not None:
            runner.stop()
        self._asked.set()

    def check_line(self, line: str) -> None:
        with self._lock:
            self.check_lines.append(line)

    def started(self, name: str, runner: "StrategyRunner") -> None:
        with self._lock:
            self.running, self.runner, self.wanted, self.error = name, runner, None, None
            self.generation += 1

    def failed(self, name: str, error: str) -> None:
        with self._lock:
            self.running, self.runner, self.wanted, self.error = None, None, None, f"{name}: {error}"
            self.generation += 1

    def take(self) -> tuple[Optional[str], bool]:
        """(the demo asked for since the last one started, whether the field check comes first);
        (None, False) when it was stopped otherwise (Ctrl+C)."""
        with self._lock:
            self._asked.clear()
            return self.wanted, self.check_wanted

    def checked(self) -> None:
        with self._lock:
            self.check_wanted = False

    def wait(self, poll_s: float = 0.5) -> tuple[Optional[str], bool]:
        """Blocks until the page asks for a demo (Ctrl+C still ends it); `take()`'s answer."""
        while not self._asked.wait(poll_s):
            pass
        return self.take()


def attach(server: DashboardServer, switcher: DemoSwitcher) -> DemoSwitcher:
    server.add_route("/demo/config", lambda: json.dumps(switcher.config()).encode())
    server.add_command_handler(
        "/demo/run", lambda payload: switcher.request(str(payload["name"]), payload.get("settings"))
    )
    server.add_command_handler("/demo/check", lambda payload: switcher.request_check())
    return switcher
