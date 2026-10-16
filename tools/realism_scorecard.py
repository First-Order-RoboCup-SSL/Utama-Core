#!/usr/bin/env python3
# ruff: noqa: E402
"""Realism scorecard: a round-robin's matches against real Division B matches, component by
component (attempts, chance quality, keeper saves, misses, ball slow-down, kick speed ...).

    # the real side, from converted official logs (`tools/convert_ssl_log.py`); rewrites the
    # committed reference (xG model + real rows)
    pixi run python tools/realism_scorecard.py --real ~/dev/ssl/official-logs/2025 ~/dev/ssl/official-logs/2026

    # a round-robin against the reference
    pixi run python tools/realism_scorecard.py --sim replays/tournament_<id>

Per-match shot records are cached: the real ones in `--real-cache`, the sim ones in
`<run_dir>/realism_shots.jsonl`; a cache is reused unless `--refresh`. What each row means and
why goal totals are not one: `utama_core/analysis/realism/scorecard.py`, `docs/sim_realism.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis.realism import scorecard

SIM_CACHE_NAME = "realism_shots.jsonl"
DEFAULT_REAL_CACHE = Path.home() / "dev/ssl/official-logs/realism_shots.jsonl"


def _replays(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        files += sorted(p.rglob("*.npz")) if p.is_dir() else [p]
    return files


def _records(paths: list[Path], cache: Path, workers: int, refresh: bool) -> list[dict]:
    if cache.exists() and not refresh:
        print(f"reusing {cache}")
        return [json.loads(line) for line in cache.open()]
    files = _replays(paths)
    out = []
    cache.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex, cache.open("w") as f:
        for i, rec in enumerate(ex.map(scorecard.analyse_replay, files)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            out.append(rec)
            print(f"[{i + 1}/{len(files)}] {rec['match']} shots={len(rec.get('shots', []))} {rec.get('error', '')}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", nargs="+", type=Path, help="converted real logs (dirs or .npz): rebuild the reference")
    p.add_argument("--real-cache", type=Path, default=DEFAULT_REAL_CACHE)
    p.add_argument("--sim-cache", type=Path, help=f"sim shot records (default <run_dir>/{SIM_CACHE_NAME})")
    p.add_argument("--sim", type=Path, help="a round-robin run dir to score against the reference")
    p.add_argument("--reference", type=Path, default=scorecard.REFERENCE_PATH)
    p.add_argument("--workers", type=int, default=1, help="processes for the replay pass (each holds a match)")
    p.add_argument("--refresh", action="store_true", help="ignore cached shot records")
    p.add_argument("--json", type=Path, help="also write the compared rows here")
    a = p.parse_args(argv)
    if not a.real and not a.sim:
        p.error("give --real, --sim or both")

    if a.real:
        records = _records(a.real, a.real_cache, a.workers, a.refresh)
        ref = scorecard.reference(records, [str(x) for x in a.real])
        a.reference.write_text(json.dumps(ref, indent=1) + "\n")
        c = ref["xg_check"]
        print(
            f"reference: {ref['matches']} matches, {ref['attempts']} attempts, {ref['goals_from_attempts']} scored; "
            f"held-out AUC {c['auc']:.3f}, log loss {c['log_loss']:.4f} vs {c['log_loss_constant']:.4f} constant"
            f" -> {a.reference}"
        )
    if a.sim:
        ref = json.loads(a.reference.read_text())
        records = _records([a.sim], a.sim_cache or a.sim / SIM_CACHE_NAME, a.workers, a.refresh)
        lines = scorecard.compare(ref, records)
        print(f"\n{a.sim.name} vs real ({ref['matches']} matches)\n")
        print(scorecard.format_table(lines))
        if a.json:
            a.json.write_text(json.dumps(lines, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
