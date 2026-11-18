"""Strategies view: every strategy in the catalog, with what it is and what we know about it, and
every evaluation run, verdicts first. Reads only what is already on disk; records nothing.

    GET /strategies       {"champion", "core", "strategies": [...]}  one entry per catalog strategy
    GET /strategies/runs  every run under `replays/`, newest first
    GET /strategies/maps?round=<run_id>  the round's position and shot maps per strategy

A strategy's entry: its description (the module docstring), file, role (champion / core / reserve,
`tools/evaluation/reference/players.json`), the tactics it imports, its verdicts (from each
`eval_*/report.json` it was a candidate in), how it plays (per-match signals in the newest round it
played, beside the average of the other strategies in that round) and its matches, each with its
replay. Names are shown as the catalog calls them today: a `.strategy` file (`name@commit.strategy`)
and a pre-rename name both show as the strategy's name, so frozen files never appear as strategies of
their own. Matches played under an older name or older code keep their date, which is how to tell.

The maps are `tools/debug/pitch_maps.py --data`, run in the background the first time a round's maps
are asked for and kept beside its summary (`pitch_maps.json`); until then the answer is "computing".
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.dashboard.server import DashboardServer
from utama_core.strategy import catalog

REPO = Path(__file__).resolve().parents[3]
PLAYERS = REPO / "tools" / "evaluation" / "reference" / "players.json"
STRATEGY_DIR = REPO / "utama_core" / "strategy"
TACTICS_PACKAGE = "utama_core.tactics"

# The signals shown per strategy, as (key, label, per): `per` is "match" (divide the round total by
# matches), "match_as_a" (real losses are measured on config_a's side only) or None (already a rate).
SIGNALS = [
    ("goals_for", "goals scored", "match"),
    ("goals_against", "goals conceded", "match"),
    ("shots", "shots", "match"),
    ("attacking_third_entries", "attacking-third entries", "match"),
    ("completed_passes", "completed passes", "match"),
    ("forward_pass_share", "forward pass share", None),
    ("pass_progress_m", "metres gained per pass", None),
    ("real_losses_as_a", "real ball losses", "match_as_a"),
    ("fouls", "fouls", "match"),
    ("stalled", "stalled matches (share)", "match"),
]


def attach(server: DashboardServer) -> None:
    server.add_route("/strategies", lambda: json.dumps(load_strategies()).encode())
    server.add_route("/strategies/runs", lambda: json.dumps(load_runs()).encode())
    server.add_route("/strategies/maps", lambda q=None: json.dumps(maps_of((q or {}).get("round", ""))).encode())


# --- names ------------------------------------------------------------------------------------


def is_frozen(config: str) -> bool:
    """A player from a `.strategy` file: recorded as its path or as `name@commit`."""
    return config.endswith(".strategy") or "@" in config


def display_name(config: str) -> str:
    """`config` as the catalog calls it today: `replays/eval/frozen/pendulum@ab12.strategy`,
    `split_shape@ab12` and `build_split_shape_kernel_strategy` all become a plain catalog name."""
    if is_frozen(config):
        config = Path(config).name.removesuffix(".strategy").split("@")[0]
    return catalog.current_name(config)


def _label(config: str) -> str:
    """How a player is shown in a match: its catalog name, marked when it is the last release's file."""
    return display_name(config) + (" (last release)" if is_frozen(config) else "")


def _newest_first(run_id: str) -> tuple:
    """A sort key: by the run's date and time (`eval_` and `tournament_` alike), then its id."""
    m = re.search(r"\d{8}_\d{6}", run_id)
    return (m[0] if m else "", run_id)


def run_date(run_id: str) -> Optional[str]:
    m = re.search(r"(\d{4})(\d{2})(\d{2})", run_id)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else None


# --- reading the disk -------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _summary_paths(base: Path) -> list[Path]:
    # older round robins: <run>/summary.json; evaluate.py: eval_<id>/round_<k>/summary.json
    return [*base.glob("*/summary.json"), *base.glob("eval_*/round_*/summary.json")]


def _summaries(base: Path) -> list[tuple[Path, dict]]:
    out = []
    for path in _summary_paths(base):
        summary = _read_json(path)
        if summary is not None:
            for result in summary.get("results", []):
                result["replay"] = _replay_path(base, path.parent, result)
            out.append((path, summary))
    out.sort(key=lambda ps: _newest_first(ps[1].get("run_id", "")), reverse=True)
    return out


def _file_name(config: str) -> str:
    """A player's name in replay file names, as `tools/evaluation/match._short_name` writes it: a
    `.strategy` file by its file name (`split_shape@cffe3a28`), anything else by its plain name."""
    return Path(config).stem if config.endswith(".strategy") else catalog.plain_name(config)


def _replay_path(base: Path, run_dir: Path, result: dict) -> Optional[str]:
    """The result's replay, relative to `base`: `evaluate.py` writes `<a>_vs_<b>.npz`. Older runs
    (the retired `ladder.py`, `full_match_tournament.py`) wrote side and kickoff tags,
    `<a>_vs_<b>_<R|L><K|k>` (as `.pkl` for the latter)."""
    stem = f"{_file_name(result['config_a'])}_vs_{_file_name(result['config_b'])}"
    names = [f"{stem}.npz", f"{stem}.pkl"]
    if "a_is_right" in result:
        names.insert(0, f"{stem}_{'R' if result['a_is_right'] else 'L'}{'K' if result.get('a_kicks_off') else 'k'}.pkl")
    for name in names:
        if (run_dir / name).exists():
            return str((run_dir / name).relative_to(base))
    return None


def _reports(base: Path) -> list[dict]:
    reports = [r for p in base.glob("eval_*/report.json") if (r := _read_json(p)) is not None]
    return sorted(reports, key=lambda r: r.get("run_id", ""), reverse=True)


# --- one strategy -----------------------------------------------------------------------------


def _docstring(path: Path) -> str:
    try:
        return ast.get_docstring(ast.parse(path.read_text())) or ""
    except (OSError, SyntaxError):
        return ""


def _first_line(text: str) -> str:
    return text.strip().split("\n\n")[0].replace("\n", " ") if text else ""


# a strategy docstring opens "`name` (until <date> `old`): what it does. The pickers only it uses are
# here."; the page already shows the name and old names
_NAME_PREFIX = re.compile(r"^`[^`]+`(\s*\(until [^)]*\))?:\s*")
_BOILERPLATE = re.compile(r"\s*[^.]*\bpickers only it uses\b[^.]*\.?")


def summary_of(doc: str) -> str:
    """What the strategy does, in one line, without its name or the docstring's boilerplate."""
    line = _BOILERPLATE.sub("", _NAME_PREFIX.sub("", _first_line(doc))).strip()
    if not line.startswith("`"):  # a strategy's name stays lower case
        line = line[:1].upper() + line[1:]
    return line.replace("`", "")


def tactics_of(path: Path) -> list[dict]:
    """The tactics a strategy file imports from `utama_core.tactics`, each with its module's first
    docstring paragraph. Read from source: importing would run the module."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return []
    found: dict[str, dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(TACTICS_PACKAGE):
            module = REPO / Path(*node.module.split("."))
            file = module.with_suffix(".py") if module.with_suffix(".py").exists() else module / "__init__.py"
            for alias in node.names:
                if alias.name[:1].isupper() and alias.name not in found:
                    found[alias.name] = {
                        "name": alias.name,
                        "file": str(file.relative_to(REPO)),
                        "description": _first_line(_docstring(file)),
                    }
    return list(found.values())


def _stalled(result: dict) -> bool:
    return bool((result.get("stats") or {}).get("stall_events") or result.get("possession_backstop"))


def _matches_of(name: str, summaries: Iterable[tuple[Path, dict]]) -> list[dict]:
    out = []
    for _, summary in summaries:
        run_id = summary.get("run_id", "")
        for r in summary.get("results", []):
            if r.get("reused"):  # listed under the run that played it
                continue
            sides = (
                (r["config_a"], r["config_b"], r["score_a"], r["score_b"]),
                (r["config_b"], r["config_a"], r["score_b"], r["score_a"]),
            )
            # today's strategy only (not the last release's file of it), and never against itself
            mine = [
                side
                for side in sides
                if display_name(side[0]) == name and not is_frozen(side[0]) and _label(side[1]) != name
            ]
            if not mine:
                continue
            _, them, score_me, score_them = mine[0]
            out.append(
                {
                    "opponent": _label(them),
                    "score": [score_me, score_them],
                    "seed": r.get("seed"),
                    "run_id": run_id,
                    "date": run_date(run_id),
                    "stalled": _stalled(r),
                    "replay": r.get("replay"),
                }
            )
    return out


def _rate(row: dict, key: str, per: Optional[str]) -> Optional[float]:
    value = row.get(key)
    if value is None:
        return None
    if per == "match":
        n = row.get("matches") or 0
    elif per == "match_as_a":
        n = row.get("matches_as_a") or 0
    else:
        return float(value)
    return value / n if n else None


MIN_SIGNAL_MATCHES = 3  # fewer is an exhibition match, not a round: its averages say little


def signals_of(name: str, summaries: Iterable[tuple[Path, dict]]) -> Optional[dict]:
    """Per-match signals in the newest round with a strategy table that `name` played at least
    `MIN_SIGNAL_MATCHES` in (else the newest it played in at all), beside the mean of the other
    strategies there. A comparison with the field, not a ranking."""
    # a frozen file is not one of today's strategies: left out of both sides of the comparison
    tables = [
        (summary, {display_name(k): v for k, v in (summary.get("strategies") or {}).items() if not is_frozen(k)})
        for _, summary in summaries
    ]
    tables = [(summary, t) for summary, t in tables if name in t]
    full = [(summary, t) for summary, t in tables if (t[name].get("matches") or 0) >= MIN_SIGNAL_MATCHES]
    for summary, table in (full or tables)[:1]:
        others = [row for n, row in table.items() if n != name]
        rows = []
        for key, label, per in SIGNALS:
            value = _rate(table[name], key, per)
            field = [v for row in others if (v := _rate(row, key, per)) is not None]
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "value": value,
                    "field": sum(field) / len(field) if field else None,
                }
            )
        run_id = summary.get("run_id", "")
        return {
            "run_id": run_id,
            "date": run_date(run_id),
            "matches": table[name].get("matches"),
            "rows": rows,
        }
    return None


def _verdicts_of(name: str, reports: Iterable[dict]) -> list[dict]:
    out = []
    for report in reports:
        for candidate, r in (report.get("candidates") or {}).items():
            if display_name(candidate) != name or "verdict" not in r:
                continue
            decided = r.get(r.get("decided_in", ""), {})
            out.append(
                {
                    "run_id": report.get("run_id"),
                    "date": run_date(report.get("run_id", "")),
                    "verdict": r["verdict"],
                    "decided_in": r.get("decided_in"),
                    "rounds": r.get("rounds"),
                    "mean": decided.get("mean"),
                    "se": decided.get("se"),
                    "p": decided.get("p"),
                    "n": decided.get("n"),
                    "reference": display_name(report.get("reference", "")),
                }
            )
    return out


def load_strategies(base: Optional[Path] = None) -> dict:
    base = Path(base or REPLAY_BASE_PATH)
    players = _read_json(PLAYERS) or {}
    champion, core = players.get("champion"), players.get("core", [])
    summaries = _summaries(base) if base.exists() else []
    reports = _reports(base) if base.exists() else []
    strategies = []
    for name in catalog.player_names():
        path = STRATEGY_DIR / f"{name}.py"
        doc = _docstring(path)
        strategies.append(
            {
                "name": name,
                "role": "champion" if name == champion else "core" if name in core else "reserve",
                "summary": summary_of(doc),
                "description": doc,
                "file": str(path.relative_to(REPO)),
                "old_names": sorted(old for old, new in catalog.RENAMED.items() if new == name),
                "tactics": tactics_of(path),
                "verdicts": _verdicts_of(name, reports),
                "signals": signals_of(name, summaries),
                "matches": _matches_of(name, summaries),
            }
        )
    return {"champion": champion, "core": core, "strategies": strategies}


# --- runs -------------------------------------------------------------------------------------


def load_runs(base: Optional[Path] = None) -> list[dict]:
    """Every run, newest first: an `evaluate.py` run is its report (if it finished) and its rounds;
    an older round robin is one round. Names in results stay as recorded, with `display_a`/`display_b`
    the catalog names; each result's `stats` is replaced by `stalled`."""
    base = Path(base or REPLAY_BASE_PATH)
    if not base.exists():
        return []
    runs: dict[str, dict] = {}
    for path, summary in _summaries(base):
        for r in summary.get("results", []):
            r["display_a"], r["display_b"] = _label(r["config_a"]), _label(r["config_b"])
            # the page needs only whether it stalled; the full stats are ~95% of the payload
            r["stalled"] = _stalled(r)
            r.pop("stats", None)
        run_dir = path.parent.parent if path.parent.name.startswith("round_") else path.parent
        run = runs.setdefault(
            run_dir.name, {"run_id": run_dir.name, "date": run_date(run_dir.name), "report": None, "rounds": []}
        )
        run["rounds"].append(summary)
    for report in _reports(base):
        run = runs.get(report.get("run_id"))
        if run is None:
            continue
        run["reference"] = display_name(report.get("reference", ""))
        run["report"] = {
            "run": report.get("run"),
            "rule": report.get("rule"),
            "failed": report.get("failed", []),
            "candidates": {
                display_name(c): {k: v for k, v in r.items() if k != "stalls"}
                for c, r in (report.get("candidates") or {}).items()
            },
        }
    for run in runs.values():
        run["rounds"].sort(key=lambda s: s.get("round", 0))
    return sorted(runs.values(), key=lambda r: _newest_first(r["run_id"]), reverse=True)


# --- maps -------------------------------------------------------------------------------------

MAPS_FILE = "pitch_maps.json"
PITCH_MAPS = REPO / "tools" / "debug" / "pitch_maps.py"
_jobs: dict[Path, subprocess.Popen] = {}
_jobs_lock = threading.Lock()


def maps_of(round_id: str, base: Optional[Path] = None, start=subprocess.Popen) -> dict:
    """A round's maps by catalog name (frozen files left out), or its status: "computing" while
    `pitch_maps.py` runs, "failed" if it ended without writing them (asking again retries), "missing"
    for no such round. `start` runs the command (a test passes a fake)."""
    base = Path(base or REPLAY_BASE_PATH).resolve()
    round_dir = (base / round_id).resolve()
    if base not in round_dir.parents or not (round_dir / "summary.json").exists():
        return {"status": "missing"}
    data = _read_json(round_dir / MAPS_FILE)
    if data is not None:
        strategies = {display_name(k): v for k, v in data.pop("strategies", {}).items() if not is_frozen(k)}
        return {"status": "ready", **data, "strategies": strategies}
    with _jobs_lock:
        job = _jobs.get(round_dir)
        if job is not None and job.poll() is not None:
            del _jobs[round_dir]
            return {"status": "failed"}
        if job is None:
            cmd = [
                sys.executable,
                str(PITCH_MAPS),
                str(round_dir),
                "--data",
                str(round_dir / MAPS_FILE),
                "--workers",
                "4",
            ]
            _jobs[round_dir] = start(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"status": "computing"}
