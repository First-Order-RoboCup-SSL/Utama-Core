"""The Live tab's Demo panel (`examples/demos.py`): a demo is switched only while halted, the
switch stops the running one, settings ride on Run, the field check only on real robots; and
every demo started (or failed) bumps the generation the page reloads on."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from utama_core.dashboard.views import demo as demo_view


def _attach(checks=False):
    routes, handlers = {}, {}
    server = SimpleNamespace(add_route=routes.__setitem__, add_command_handler=handlers.__setitem__)
    switcher = demo_view.attach(
        server, demo_view.DemoSwitcher(["drill", "pass"], {"camera": ["full", "left"]}, checks=checks)
    )
    return switcher, lambda: json.loads(routes["/demo/config"]()), handlers["/demo/run"], handlers["/demo/check"]


def _runner(halted):
    runner = SimpleNamespace(halted=halted, stopped=0)
    runner.stop = lambda: setattr(runner, "stopped", runner.stopped + 1)
    return runner


def test_a_switch_while_halted_stops_the_running_demo_and_names_the_next():
    switcher, config, run, _ = _attach()
    runner = _runner(halted=True)
    switcher.started("drill", runner)
    run({"name": "pass", "settings": {"camera": "left"}})
    assert runner.stopped == 1
    c = config()
    assert (c["running"], c["switching"], c["generation"]) == ("drill", "pass", 1)
    assert c["settings"] == {"camera": {"values": ["full", "left"], "value": "left"}}
    with pytest.raises(ValueError, match="already"):  # a second click before it starts
        run({"name": "drill"})
    assert switcher.take() == ("pass", False)
    switcher.started("pass", _runner(halted=True))
    assert config()["running"] == "pass" and config()["switching"] is None and config()["generation"] == 2


def test_a_switch_in_play_or_to_an_unknown_demo_or_setting_is_refused():
    switcher, config, run, _ = _attach()
    runner = _runner(halted=False)
    switcher.started("drill", runner)
    with pytest.raises(ValueError, match="Halt"):
        run({"name": "pass"})
    assert runner.stopped == 0 and switcher.take() == (None, False)  # stopped by Ctrl+C now: the script ends
    runner.halted = True
    with pytest.raises(ValueError, match="no demo"):
        run({"name": "nope"})
    with pytest.raises(ValueError, match="no setting"):
        run({"name": "pass", "settings": {"camera": "fisheye"}})
    assert config()["settings"]["camera"]["value"] == "full" and runner.stopped == 0


def test_the_field_check_stops_the_demo_then_asks_for_it_again_and_only_on_real_robots():
    switcher, config, _, check = _attach(checks=True)
    runner = _runner(halted=True)
    switcher.started("drill", runner)
    check({})
    assert runner.stopped == 1 and config()["checking"] is True
    assert switcher.take() == ("drill", True)
    switcher.check_line("ok camera 0: 60 packets/s")
    switcher.checked()
    assert config()["checking"] is False and config()["check_lines"] == ["ok camera 0: 60 packets/s"]

    _, _, _, sim_check = _attach(checks=False)
    with pytest.raises(ValueError, match="real robots"):
        sim_check({})


def test_a_demo_that_failed_to_start_shows_why_and_any_demo_can_be_picked():
    switcher, config, run, _ = _attach()
    switcher.failed("drill", "radio 1 not found")
    assert config()["error"] == "drill: radio 1 not found" and config()["running"] is None
    threading.Timer(0.05, run, args=({"name": "pass"},)).start()  # no runner to halt: allowed
    assert switcher.wait(poll_s=0.01) == ("pass", False)
