"""`examples/demos.py`: plays the first demo, then each the page asks for, one runner at a time
(the last closed before the next is built, so it can take the radio); a demo that fails to start
waits for another pick; stopped otherwise (Ctrl+C), it ends."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import demos as ex
finally:
    sys.path.remove(str(EXAMPLES))

from utama_core.dashboard.views.demo import DemoSwitcher


class _Runner:
    def __init__(self, name, log, then=None):
        self.name, self.log, self.then, self.halted = name, log, then, True

    def stop(self):
        self.log.append(f"stop {self.name}")

    def run(self):
        self.log.append(f"run {self.name}")
        if self.then:
            self.then()  # what the page does while this demo plays
        self.log.append(f"closed {self.name}")


def test_each_demo_asked_for_is_built_after_the_last_closed_and_ctrl_c_ends_it():
    log, switcher = [], DemoSwitcher(["a", "b"])
    demos = {
        "a": lambda mode, server: (log.append("build a"), _Runner("a", log, then=lambda: switcher.request("b")))[1],
        "b": lambda mode, server: (log.append("build b"), _Runner("b", log))[1],  # its run ends as Ctrl+C would
    }
    ex.play("rsim", "a", server=None, switcher=switcher, demos=demos)
    assert log == ["build a", "run a", "stop a", "closed a", "build b", "run b", "closed b"]
    assert switcher.config()["running"] == "b" and switcher.config()["generation"] == 2


def test_a_demo_that_fails_to_start_waits_for_the_next_pick():
    log, switcher = [], DemoSwitcher(["a", "b"])

    def broken(mode, server):
        threading.Timer(0.05, switcher.request, args=("b",)).start()  # the page picks "b" while it waits
        raise SystemExit("radio 1 not found")

    demos = {"a": broken, "b": lambda mode, server: _Runner("b", log)}
    ex.play("real", "a", server=None, switcher=switcher, demos=demos)
    assert log == ["run b", "closed b"]
