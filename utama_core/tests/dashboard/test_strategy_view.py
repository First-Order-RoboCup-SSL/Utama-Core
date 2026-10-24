"""The Live tab's Strategy panel: what each side may play, and Run handing a fresh strategy to
the runner, which refuses unless halted."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from utama_core.dashboard.views import strategy as strategy_view


def _attach(replace):
    routes, handlers = {}, {}
    server = SimpleNamespace(add_route=routes.__setitem__, add_command_handler=handlers.__setitem__)
    runner = SimpleNamespace(my_team_is_yellow=False, replace_strategy=replace)
    built = []
    choices = {
        "my": {"lead": lambda: built.append("lead") or "lead", "press": lambda: built.append("press") or "press"},
        "opp": {"goalkeeper": lambda: "goalkeeper"},
    }
    strategy_view.attach(server, runner, choices, running={"my": "lead", "opp": "goalkeeper"})
    return routes["/strategy/config"], handlers["/strategy/run"], built


def test_run_builds_a_fresh_strategy_for_the_runner_and_the_config_shows_it_running():
    replaced = []
    config, run, built = _attach(lambda side, s: replaced.append((side, s)))
    run({"side": "my", "name": "press"})
    run({"side": "my", "name": "press"})  # a restart: built afresh again
    assert replaced == [("my", "press"), ("my", "press")] and built == ["press", "press"]
    assert json.loads(config()) == {
        "my": {"team": "blue", "choices": ["lead", "press"], "running": "press"},
        "opp": {"team": "yellow", "choices": ["goalkeeper"], "running": "goalkeeper"},
    }


def test_a_refused_run_keeps_what_is_shown_running():
    def refuse(side, s):
        raise ValueError("press Halt first")

    config, run, _ = _attach(refuse)
    with pytest.raises(ValueError, match="Halt"):
        run({"side": "my", "name": "press"})
    with pytest.raises(ValueError, match="no strategy"):
        run({"side": "opp", "name": "press"})
    assert json.loads(config())["my"]["running"] == "lead"
