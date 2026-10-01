"""Stored round-robin match results, reused while the code a match runs is unchanged.

An rsim match is deterministic, so a result is a function of `fingerprint.match_key`:
both sides' code, the shared code and environment, and the run settings. A record holds
everything `smoke_tournament.py` reads from a played match: the result and its stats, the
restart episodes (`restart_outcomes.analyse_match`) and the ball-loss record
(`turnover_breakdown.analyse_match`). No replay is stored; `--pair A B` plays any match
again, byte for byte.

A fingerprint can miss a dependency (see `fingerprint.py`'s "Known gaps"), so a run also
replays a sample of the matches it would reuse (`spot_check_sample`) and compares the
fresh record with the stored one (`differences`).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional

from utama_core.config.settings import REPLAY_BASE_PATH

CACHE_DIR = Path(REPLAY_BASE_PATH) / "match_cache"
RECORD_PARTS = ("result", "restarts", "losses")


def _plain(value):
    """`value` as it reads back from JSON (tuples become lists, keys become strings)."""
    return json.loads(json.dumps(value))


class MatchCache:
    def __init__(self, root: Path = CACHE_DIR):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        try:
            return json.loads(self.path(key).read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, record: dict) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_plain(record), sort_keys=True))
        tmp.replace(path)

    def evict(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)


def spot_check_sample(keys: Iterable[str], fraction: float, seed: int = 0) -> set[str]:
    """The cached keys a run replays to check them: `fraction` of them, at least one when
    there are any and `fraction` > 0. Seeded, so a rerun checks the same matches."""
    keys = sorted(keys)
    if not keys or fraction <= 0:
        return set()
    n = min(len(keys), max(1, round(fraction * len(keys))))
    return set(random.Random(seed).sample(keys, n))


def differences(stored: dict, fresh: dict) -> list[str]:
    """The record parts where a replayed match disagrees with its stored record."""
    return [part for part in RECORD_PARTS if _plain(stored.get(part)) != _plain(fresh.get(part))]
