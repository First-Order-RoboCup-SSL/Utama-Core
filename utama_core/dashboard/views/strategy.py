"""Strategy view — choose, or restart, what each side of a running match plays, from the Live tab.

The script that builds the runner says what can be chosen: for each side ("my", "opp"), names
mapped to functions that build a fresh `AbstractStrategy`, and the name it started with. Running a
choice (the one running restarts it) hands a new strategy to `StrategyRunner.replace_strategy`,
which takes it at the next tick and only while HALTed; a refusal comes back to the page as a 400.
When the run records a game log, a note box writes the operator's notes into it (`GameLogWriter.note`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Dict

from utama_core.dashboard.server import DashboardServer

if TYPE_CHECKING:
    from utama_core.engine.abstract_strategy import AbstractStrategy
    from utama_core.run import StrategyRunner

Choices = Dict[str, Callable[[], "AbstractStrategy"]]


def attach(
    server: DashboardServer,
    runner: "StrategyRunner",
    choices: Dict[str, Choices],
    running: Dict[str, str],
) -> None:
    """`choices`: side ("my"/"opp") -> {name: build a fresh strategy}; `running`: side -> the
    name each side started with."""
    running = dict(running)

    def _config_bytes() -> bytes:
        yellow = {"my": runner.my_team_is_yellow, "opp": not runner.my_team_is_yellow}
        return json.dumps(
            {
                "sides": {
                    side: {
                        "team": "yellow" if yellow[side] else "blue",
                        "choices": list(names),
                        "running": running[side],
                    }
                    for side, names in choices.items()
                },
                "notes": runner.game_log is not None,
            }
        ).encode()

    def _handle_run(payload: dict) -> None:
        side, name = payload["side"], payload["name"]
        if side not in choices or name not in choices[side]:
            raise ValueError(f"no strategy {name!r} for {side!r}")
        runner.replace_strategy(side, choices[side][name](), name)
        running[side] = name

    def _handle_note(payload: dict) -> None:
        text = str(payload.get("text", "")).strip()
        if runner.game_log is None:
            raise ValueError("this run records no game log")
        if text:
            runner.game_log.note(text)

    server.add_route("/strategy/config", _config_bytes)
    server.add_command_handler("/strategy/run", _handle_run)
    server.add_command_handler("/log/note", _handle_note)
