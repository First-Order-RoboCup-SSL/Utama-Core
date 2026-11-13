"""Every match result the evaluation tools have played, in one store: `replays/match_results/`.

An rsim match is deterministic, so its result is a function of its fixture: the environment (the
shared code and machine, the match length and the sim noise without its seed), both players' code fingerprints (`fingerprint.py`), their sides and the noise
seed. `tools/evaluation/evaluate.py`'s `play_and_record` plays a fixture and returns its record;
`evaluate.py` looks a fixture up first and plays it only when the store
does not hold it, then stores the record. Reuse is automatic: there is no flag. A player whose code
changed has new fingerprints, so only its matches play again.

Layout: one JSON file per fixture, under its environment: `replays/match_results/<env>/<fixture>.json`.
A record holds the score, the shots, the stall events, the full per-match stats, the restart
episodes and the ball-loss record of the match, the names of both players as labels, the run it came
from, and the fingerprints it was played with. Names are labels only: `load(env, fps)` relabels a
player by its fingerprint, so a `.strategy` file and the same code in the repo are one player.

To force replays, delete the store, or one environment's folder inside it. The store does not check
itself: if a fingerprint misses a dependency (see `fingerprint.py`, "Known gaps"), the stale results
stay until the store is deleted. The earlier stores (`replays/match_cache/`, `replays/eval/ledger.jsonl`)
are no longer read or written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from utama_core.replay.fingerprint import MATCH_ENTRY, CodeGraph, _digest

STORE_NAME = "match_results"


def open_store(replay_base: Path) -> "MatchResults":
    """The store under a replay base directory (`REPLAY_BASE_PATH` for the tools)."""
    return MatchResults(Path(replay_base) / STORE_NAME)


def environment(
    graph: CodeGraph,
    *,
    duration_seconds: float,
    sim_noise: Optional[dict],
) -> str:
    """Id of what every match of one evaluation or round-robin shares. `sim_noise` is
    `SimNoise.to_dict()`, its seed left out (the seed is part of the fixture)."""
    noise = {k: v for k, v in (sim_noise or {}).items() if k != "seed"}
    settings = {
        "duration_seconds": duration_seconds,
        "sim_noise": noise,
    }
    return _digest(
        [
            "evaluation environment",
            graph.base_fingerprint([MATCH_ENTRY]),
            json.dumps(settings, sort_keys=True),
        ]
    )[:16]


def fixture_id(
    env: str,
    a_fp: str,
    b_fp: str,
    seed: Optional[int],
    *,
    a_is_right: bool = True,
    a_kicks_off: bool = True,
) -> str:
    """One match: environment, both players' fingerprints, the sides and the noise seed. The sides
    default to the tools' convention (config_a yellow, on the right, kicking off)."""
    sides = {"a_is_right": a_is_right, "a_kicks_off": a_kicks_off}
    return _digest([env, a_fp, b_fp, json.dumps(seed), json.dumps(sides, sort_keys=True)])[:16]


def stall_summary(events: Optional[Iterable[dict]]) -> Optional[list[dict]]:
    """A match's stall events as stored rows (kind, sim time, and the diagnosis if there is one),
    or None when the match's stats were not collected: unknown, which is not "no stalls".
    The watchdog only watches config_a, so these are config_a's stalls."""
    if events is None:
        return None
    rows = []
    for e in events:
        row = {"kind": e["kind"], "sim_time": e["sim_time"]}
        if e.get("diagnosis"):
            row["diagnosis"] = e["diagnosis"]
        rows.append(row)
    return rows


def record(
    env: str,
    a_fp: str,
    b_fp: str,
    result: dict,
    *,
    seed: Optional[int],
    source: str,
    restarts: Optional[list[dict]],
    losses: Optional[dict],
    a_is_right: bool = True,
    a_kicks_off: bool = True,
    replay: Optional[str] = None,
) -> dict:
    """The stored record of one played match. `result` is a `match.MatchResult` as a dict; `restarts`
    and `losses` are the replay analyses (None when the replay was not kept to analyse). `replay` is
    where its files are, a stem relative to the replay base directory (`eval_<id>/round_<k>/a_vs_b`),
    so a later run that reuses the match can link them into its own round folder."""
    stats = result.get("stats") or {}
    shots = stats.get("shots") or {}
    rec = {
        "fixture": fixture_id(env, a_fp, b_fp, seed, a_is_right=a_is_right, a_kicks_off=a_kicks_off),
        "env": env,
        "a": result["config_a"],
        "a_fp": a_fp,
        "b": result["config_b"],
        "b_fp": b_fp,
        "a_is_right": a_is_right,
        "a_kicks_off": a_kicks_off,
        "seed": seed,
        "source": source,
        "score_a": result["score_a"],
        "score_b": result["score_b"],
        "shots_a": shots.get("friendly"),
        "shots_b": shots.get("enemy"),
        "stalls": stall_summary(stats.get("stall_events") if stats else None),
        "stats": result.get("stats"),
        "restarts": restarts,
        "losses": losses,
        "replay": replay,
    }
    return json.loads(json.dumps(rec))  # the shape it reads back in, tuples as lists


class MatchResults:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, env: str, fixture: str) -> Path:
        return self.root / env / f"{fixture}.json"

    def get(self, env: str, fixture: str) -> Optional[dict]:
        try:
            return json.loads(self.path(env, fixture).read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, rec: dict) -> None:
        path = self.path(rec["env"], rec["fixture"])
        path.parent.mkdir(parents=True, exist_ok=True)
        # One temp file per process: several runs (one per strategy branch) can share the store and
        # store the same match at once; a shared temp file could be renamed away mid-write.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        tmp.replace(path)

    def load(self, env: str, fps: Optional[dict[str, str]] = None) -> dict[str, dict]:
        """The stored matches of `env`, by fixture id. With `fps` (name -> fingerprint), a player it
        knows goes by that name, whatever the name was when the match was played."""
        names = {fp: name for name, fp in (fps or {}).items()}
        out: dict[str, dict] = {}
        folder = self.root / env
        if not folder.is_dir():
            return out
        for path in sorted(folder.glob("*.json")):
            try:
                rec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue  # a torn write from a crash, or a file someone edited: not a result
            rec["a"], rec["b"] = names.get(rec["a_fp"], rec["a"]), names.get(rec["b_fp"], rec["b"])
            out[rec["fixture"]] = rec
        return out
