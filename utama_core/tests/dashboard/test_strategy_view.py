"""The Live tab's Strategy panel: what each side may play, and Run handing a fresh strategy to
the runner, which refuses unless halted."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from utama_core.dashboard.views import strategy as strategy_view


def _attach(replace, game_log=None):
    routes, handlers = {}, {}
    server = SimpleNamespace(add_route=routes.__setitem__, add_command_handler=handlers.__setitem__)
    runner = SimpleNamespace(my_team_is_yellow=False, replace_strategy=replace, game_log=game_log)
    built = []
    choices = {
        "my": {"lead": lambda: built.append("lead") or "lead", "press": lambda: built.append("press") or "press"},
        "opp": {"goalkeeper": lambda: "goalkeeper"},
    }
    strategy_view.attach(server, runner, choices, running={"my": "lead", "opp": "goalkeeper"})
    return routes["/strategy/config"], handlers["/strategy/run"], built, handlers["/log/note"]


def test_run_builds_a_fresh_strategy_for_the_runner_and_the_config_shows_it_running():
    replaced = []
    config, run, built, _ = _attach(lambda side, s, name: replaced.append((side, s, name)))
    run({"side": "my", "name": "press"})
    run({"side": "my", "name": "press"})  # a restart: built afresh again
    assert replaced == [("my", "press", "press"), ("my", "press", "press")] and built == ["press", "press"]
    assert json.loads(config()) == {
        "sides": {
            "my": {"team": "blue", "choices": ["lead", "press"], "running": "press"},
            "opp": {"team": "yellow", "choices": ["goalkeeper"], "running": "goalkeeper"},
        },
        "notes": False,
    }


def test_a_refused_run_keeps_what_is_shown_running():
    def refuse(side, s, name):
        raise ValueError("press Halt first")

    config, run, _, _ = _attach(refuse)
    with pytest.raises(ValueError, match="Halt"):
        run({"side": "my", "name": "press"})
    with pytest.raises(ValueError, match="no strategy"):
        run({"side": "opp", "name": "press"})
    assert json.loads(config())["sides"]["my"]["running"] == "lead"


def test_a_note_goes_into_the_game_log_and_needs_one():
    notes = []
    config, _, _, note = _attach(lambda *a: None, game_log=SimpleNamespace(note=notes.append))
    note({"text": "  ball placed by hand "})
    note({"text": ""})  # an empty box writes nothing
    assert notes == ["ball placed by hand"] and json.loads(config())["notes"] is True
    _, _, _, note = _attach(lambda *a: None)
    with pytest.raises(ValueError, match="no game log"):
        note({"text": "x"})
