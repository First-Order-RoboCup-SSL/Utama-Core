"""Every evaluation match's result, in one append-only file, for ratings across runs.

`tools/evaluation/evaluate.py` and `round_robin.py` append one JSON line per match they play:
the environment it ran in, both sides with the fingerprint of their code, the noise seed, the
score and the shots. A rating fit reads back the matches of one environment, so each result
counts in every later evaluation until the code that played it changes.

- **Environment** (`environment`): the shared code and machine (`fingerprint.base_fingerprint`
  of the match runner), the match length, the planner and the sim noise without its seed.
  Matches from different environments are never mixed: a planner or sim change replays all.
- **Player**: a strategy name and its `fingerprint.strategy_fingerprint`. Editing a strategy
  makes it a new player; its old matches stay in the file but no longer count.
- **Fixture** (`fixture_id`): environment, both players, sides and seed. An rsim match is a
  pure function of it, so a fixture already in the ledger is never played again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.replay.fingerprint import ROUND_ROBIN_ENTRY, CodeGraph, _digest

LEDGER_PATH = Path(REPLAY_BASE_PATH) / "eval" / "ledger.jsonl"


def environment(
    graph: CodeGraph,
    *,
    duration_seconds: float,
    sim_noise: Optional[dict],
    control_scheme: str = "fpp",
) -> str:
    """Id of what every match of an evaluation shares; `sim_noise` is `SimNoise.to_dict()`,
    its seed left out."""
    noise = {k: v for k, v in (sim_noise or {}).items() if k != "seed"}
    return _digest(
        [
            "evaluation environment",
            graph.base_fingerprint([ROUND_ROBIN_ENTRY]),
            json.dumps(
                {"duration_seconds": duration_seconds, "control_scheme": control_scheme, "sim_noise": noise},
                sort_keys=True,
            ),
        ]
    )[:16]


def fixture_id(env: str, a: str, a_fp: str, b: str, b_fp: str, seed: Optional[int]) -> str:
    """config_a is always yellow and starts on the right, kicking off (`match.run_match`'s defaults)."""
    return _digest([env, a, a_fp, b, b_fp, json.dumps(seed)])[:16]


def entry(env: str, graph: CodeGraph, result: dict, seed: Optional[int], source: str) -> dict:
    """A ledger line for `result` (a `match.MatchResult` as a dict, with its stats)."""
    a, b = result["config_a"], result["config_b"]
    a_fp, b_fp = graph.strategy_fingerprint(a)[:16], graph.strategy_fingerprint(b)[:16]
    shots = (result.get("stats") or {}).get("shots") or {}
    return {
        "fixture": fixture_id(env, a, a_fp, b, b_fp, seed),
        "env": env,
        "a": a,
        "a_fp": a_fp,
        "b": b,
        "b_fp": b_fp,
        "seed": seed,
        "score_a": result["score_a"],
        "score_b": result["score_b"],
        "shots_a": shots.get("friendly"),
        "shots_b": shots.get("enemy"),
        "source": source,
    }


def append(entries: Iterable[dict], path: Path = LEDGER_PATH) -> None:
    """One line per entry; each write is a single append, so concurrent writers interleave
    whole lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")
            f.flush()


def load(env: str, path: Path = LEDGER_PATH) -> dict[str, dict]:
    """The ledger's entries for `env`, by fixture id (the last one wins if a fixture was played
    twice)."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a crash
        if e.get("env") == env:
            out[e["fixture"]] = e
    return out
