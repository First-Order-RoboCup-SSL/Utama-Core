"""The Live tab's Demo panel (`examples/demos.py`): a demo is switched only while halted, the
switch stops the running one, and every demo started (or failed) bumps the generation the page
reloads on."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from utama_core.dashboard.views import demo as demo_view


def _attach():
    routes, handlers = {}, {}
    server = SimpleNamespace(add_route=routes.__setitem__, add_command_handler=handlers.__setitem__)
    switcher = demo_view.attach(server, demo_view.DemoSwitcher(["drill", "pass"]))
    return switcher, lambda: json.loads(routes["/demo/config"]()), handlers["/demo/run"]


def _runner(halted):
    runner = SimpleNamespace(halted=halted, stopped=0)
    runner.stop = lambda: setattr(runner, "stopped", runner.stopped + 1)
    return runner


def test_a_switch_while_halted_stops_the_running_demo_and_names_the_next():
    switcher, config, run = _attach()
    runner = _runner(halted=True)
    switcher.started("drill", runner)
    run({"name": "pass"})
    assert runner.stopped == 1
    assert config() == {
        "choices": ["drill", "pass"],
        "running": "drill",
        "switching": "pass",
        "error": None,
        "generation": 1,
    }
    assert switcher.take() == "pass"
    switcher.started("pass", _runner(halted=True))
    assert config()["running"] == "pass" and config()["switching"] is None and config()["generation"] == 2


def test_a_switch_in_play_is_refused_and_the_demo_plays_on():
    switcher, config, run = _attach()
    runner = _runner(halted=False)
    switcher.started("drill", runner)
    with pytest.raises(ValueError, match="Halt"):
        run({"name": "pass"})
    assert runner.stopped == 0 and switcher.take() is None  # stopped by Ctrl+C now: the script ends
    with pytest.raises(ValueError, match="no demo"):
        run({"name": "nope"})


def test_a_demo_that_failed_to_start_shows_why_and_any_demo_can_be_picked():
    switcher, config, run = _attach()
    switcher.failed("drill", "radio 1 not found")
    assert config()["error"] == "drill: radio 1 not found" and config()["running"] is None
    threading.Timer(0.05, run, args=({"name": "pass"},)).start()  # no runner to halt: allowed
    assert switcher.wait(poll_s=0.01) == "pass"
