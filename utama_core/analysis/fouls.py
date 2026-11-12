"""Our side's fouls in a round-robin run, with the referee command active at each: the starting points for
new regression moments. `tools/find_bugs.py --fouls` prints them; `tools/moments.py` uses `restart_side` in
its checks. (Until 2026-11-11 this was `moments.py fouls`.)

`out_of_bounds`, the ball leaving the field, is a stoppage rather than a rules foul (`STOPPAGE_RULES`).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.scenario import _referee_command_at
from utama_core.strategy.catalog import plain_name

REPO_ROOT = Path(__file__).resolve().parents[2]

# Rules that stop play without being a rules foul: the ball leaving the field.
STOPPAGE_RULES = frozenset({"out_of_bounds"})
FOUL_COLUMNS = ("strategy", "rule", "tactic", "restart")


def restart_side(command: Optional[RefereeCommand], we_are_yellow: bool = True) -> Optional[str]:
    """ "ours" for a restart our team takes, "theirs" for one the other team takes, None for
    STOP, live play and timeouts."""
    if command is None or command.name.startswith("TIMEOUT"):
        return None
    if command.name.endswith("YELLOW" if we_are_yellow else "BLUE"):
        return "ours"
    if command.name.endswith("BLUE" if we_are_yellow else "YELLOW"):
        return "theirs"
    return None


def without_own_restarts(fouls: list[dict]) -> list[dict]:
    """`fouls` minus those during a restart our team takes: `keeps_clear` does not apply there."""
    return [f for f in fouls if f.get("restart") != "ours"]


def without_stoppages(fouls: list[dict]) -> list[dict]:
    return [f for f in fouls if f["rule"] not in STOPPAGE_RULES]


def _foul_column(f: dict, column: str) -> str:
    if column == "strategy":
        return plain_name(f["config_a"])
    if column == "restart":
        return f.get("restart") or "live"
    return str(f.get(column) or "-")


def count_fouls(fouls: list[dict], columns: list[str]) -> list[tuple[tuple[str, ...], int]]:
    """How many of `fouls` fall in each combination of `columns` (names from `FOUL_COLUMNS`),
    most common first; ties in name order."""
    counts = Counter(tuple(_foul_column(f, c) for c in columns) for f in fouls)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def fouls_json(fouls: list[dict], columns: Optional[list[str]] = None) -> str:
    """`fouls` as JSON: the foul dicts as they are, or with `columns` the `count_fouls` rows as
    {column: value, ..., "count": n} objects, most common first."""
    if columns is None:
        data = fouls
    else:
        data = [{**dict(zip(columns, key)), "count": n} for key, n in count_fouls(fouls, columns)]
    return json.dumps(data, indent=1, sort_keys=True)


def fouls_by_table(fouls: list[dict], columns: list[str]) -> list[str]:
    """`count_fouls` as table lines: a count column, then one column per name, then the total."""
    rows = count_fouls(fouls, columns)
    widths = [max([len(c)] + [len(key[i]) for key, _ in rows]) for i, c in enumerate(columns)]
    lines = [f"{'count':>5}  " + "  ".join(f"{c:<{w}}" for c, w in zip(columns, widths))]
    for key, n in rows:
        lines.append(f"{n:>5}  " + "  ".join(f"{v:<{w}}" for v, w in zip(key, widths)))
    lines.append(f"{sum(n for _, n in rows):>5}  total")
    return lines


def foul_line(f: dict) -> str:
    """One foul as a line: rule, time, robot, tactic, referee command, restart side, match."""
    return (
        f"{f['rule']:<28} t={f['sim_time']:>7.2f}  robot {f.get('robot_id')}  {f.get('tactic') or '':<24} "
        f"{(f['command'] or '?'):<22} {f['restart'] or 'live':<7} "
        f"{plain_name(f['config_a'])} vs {plain_name(f['config_b'])}"
    )


def _repo_relative(path: Path) -> str:
    """`path` relative to the repo root (as a string), or as given if it is outside the repo."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def our_fouls(run_dir: Path, rule: Optional[str] = None) -> list[dict]:
    """Every foul config_a (our side in the replay) committed in a round-robin run, from its
    summary.json. Each carries the referee `command` active at its time (from the match's
    `.intentions.jsonl`, None if absent), its `restart` side ("ours", "theirs", or None for live play
    and STOP) and its match's `replay` (the .npz, else the .pkl, relative to the repo root; None if
    neither is there), for replay_trace and repro."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    out = []
    for r in summary["results"]:
        stem = run_dir / f"{plain_name(r['config_a'])}_vs_{plain_name(r['config_b'])}"
        sidecar = Path(f"{stem}.intentions.jsonl")
        replay = next((p for p in (Path(f"{stem}.npz"), Path(f"{stem}.pkl")) if p.exists()), None)
        for f in (r.get("stats") or {}).get("fouls") or []:
            if f.get("side") == "friendly" and (rule is None or f["rule"] == rule):
                command = _referee_command_at(sidecar, f["sim_time"])
                out.append(
                    {
                        "config_a": r["config_a"],
                        "config_b": r["config_b"],
                        **f,
                        "command": command.name if command else None,
                        "restart": restart_side(command),
                        "replay": _repo_relative(replay) if replay is not None else None,
                    }
                )
    return out
