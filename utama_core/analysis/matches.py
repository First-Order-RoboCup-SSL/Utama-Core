"""The matches in a run dir: which two strategies a replay is between, and the score it ended on.

A match file's stem is `<config_a>_vs_<config_b>` (round-robin) or `<config_a>_vs_<config_b>_s<seed>` (an
evaluate.py run). config_a is the friendly side (yellow) in every run. Shared by `tools/evaluation/explain.py`
and `tools/triage.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utama_core.replay.columnar_reader import load_columnar_replay

STEM = re.compile(r"^(?P<a>.+?)_vs_(?P<b>.+?)(?:_s(?P<seed>\d+))?$")


@dataclass
class Match:
    """One replay in the run dir. `a` is config_a (the friendly side), `b` config_b (enemy)."""

    npz: Path
    a: str  # short names
    b: str
    seed: Optional[int]
    score_a: Optional[int]  # None when the replay has no referee state to read a score from
    score_b: Optional[int]
    fouls: list[dict]

    def job(self) -> tuple[str, str, str, list[dict]]:
        return (str(self.npz), self.a, self.b, self.fouls)

    def goal_difference(self, name: str) -> Optional[int]:
        if self.score_a is None:
            return None
        return self.score_a - self.score_b if name == self.a else self.score_b - self.score_a


def final_score(npz: Path) -> tuple[Optional[int], Optional[int]]:
    """The score at the end of a replay: the last referee state the writer stored (scores only
    change through stored referee messages). Yellow is config_a (friendly) in every run."""
    replay = load_columnar_replay(npz)
    if not replay.sparse_referee:
        return None, None
    ref = replay.sparse_referee[max(replay.sparse_referee)]
    return int(ref.yellow_team.score), int(ref.blue_team.score)


def load_match(npz: Path) -> Optional[Match]:
    """The `Match` for one replay file, or None when its name is not `<a>_vs_<b>[_s<seed>]`."""
    m = STEM.match(npz.name.removesuffix(".npz"))
    if m is None:
        return None
    stats = npz.with_name(npz.name.removesuffix(".npz") + ".stats.json")
    fouls = []
    if stats.exists():
        fouls = json.loads(stats.read_text()).get("fouls") or []
    score_a, score_b = final_score(npz)
    seed = int(m["seed"]) if m["seed"] is not None else None
    return Match(npz, m["a"], m["b"], seed, score_a, score_b, fouls)


def find_matches(run_dir: Path, names: set[str]) -> list[Match]:
    """Every replay in `run_dir` with `names` (short names) on either side, in file-name order."""
    matches = []
    for npz in sorted(run_dir.glob("*.npz")):
        m = STEM.match(npz.name.removesuffix(".npz"))
        if m is None or not ({m["a"], m["b"]} & names):
            continue
        matches.append(load_match(npz))
    return matches
