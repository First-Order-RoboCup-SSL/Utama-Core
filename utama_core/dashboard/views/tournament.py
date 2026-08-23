"""Tournament view — reads `replays/<run_id>/summary.json` written by
`full_match_tournament.py` and exposes it to the dashboard.

Fully standalone: no coupling to a running match or `StrategyRunner`. Just a
`GET /tournament/runs` route returning every run's full summary inline
(newest first), since the run count is small and this avoids needing a
second dynamic-path route on `DashboardServer`.
"""

from __future__ import annotations

import json
from typing import List

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.dashboard.server import DashboardServer


def attach(server: DashboardServer) -> None:
    server.add_route("/tournament/runs", _runs_bytes)


def _runs_bytes() -> bytes:
    return json.dumps(_load_runs()).encode()


def _load_runs() -> List[dict]:
    if not REPLAY_BASE_PATH.exists():
        return []

    summaries = []
    for summary_path in REPLAY_BASE_PATH.glob("*/summary.json"):
        try:
            with open(summary_path) as f:
                summaries.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue

    summaries.sort(key=lambda s: s.get("run_id", ""), reverse=True)
    return summaries
