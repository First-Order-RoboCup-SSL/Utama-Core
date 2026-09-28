"""Tournament view — reads `replays/<run_id>/summary.json` written by
`full_match_tournament.py` and exposes it to the dashboard.

Fully standalone: no coupling to a running match or `StrategyRunner`. Just a
`GET /tournament/runs` route returning every run's full summary inline
(newest first), since the run count is small and this avoids needing a
second dynamic-path route on `DashboardServer`. Each result gains `replay`:
its replay's path under `REPLAY_BASE_PATH`, or None when there is no file, so
the page links only to replays that exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

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
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for result in summary.get("results", []):
            result["replay"] = _replay_path(summary_path.parent, result)
        summaries.append(summary)

    summaries.sort(key=lambda s: s.get("run_id", ""), reverse=True)
    return summaries


def _short(config: str) -> str:
    return config.removeprefix("build_").removesuffix("_kernel_strategy")


def _replay_path(run_dir: Path, result: dict) -> Optional[str]:
    """The result's replay, relative to `REPLAY_BASE_PATH`: `smoke_tournament.py` writes
    `<a>_vs_<b>.npz`, `full_match_tournament.py` `<a>_vs_<b>_<R|L><K|k>.pkl`."""
    stem = f"{_short(result['config_a'])}_vs_{_short(result['config_b'])}"
    names = [f"{stem}.npz", f"{stem}.pkl"]
    if "a_is_right" in result:
        names.insert(0, f"{stem}_{'R' if result['a_is_right'] else 'L'}{'K' if result.get('a_kicks_off') else 'k'}.pkl")
    for name in names:
        if (run_dir / name).exists():
            return str((run_dir / name).relative_to(REPLAY_BASE_PATH))
    return None
