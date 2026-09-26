"""jev_summary.py — aggregate every `<tag>.result.json` in a jev_tournament run directory.

`jev_tournament.py` writes `summary.json` from the matches *its own process* played, so when several
workers share one `--out-dir` (see `tools/jev_full_round.sh`) each overwrites the others'. This reads
the per-match result files instead, so it is correct however the run was split, and can be run while
the run is still going.

Usage:
    pixi run python -m tools.jev_summary replays/jev_full_600s [--expect 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CELL_ORDER = ["_Rk", "_R", "_Lk", "_L"]


def main() -> None:
    ap = argparse.ArgumentParser(description="aggregate jev_tournament result.json files")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--expect", type=int, default=4, help="cells per opponent (for the 'missing' column)")
    args = ap.parse_args()

    results = [json.loads(p.read_text()) for p in sorted(args.run_dir.glob("*.result.json"))]
    if not results:
        raise SystemExit(f"no *.result.json in {args.run_dir}")

    table: dict[str, dict] = {}
    for r in results:
        row = table.setdefault(r["opponent"], {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "cells": {}})
        row[{"win": "W", "draw": "D", "loss": "L"}[r["outcome"]]] += 1
        row["GF"] += r["score_openjev"]
        row["GA"] += r["score_opponent"]
        cell = "_" + r["tag"].rsplit("_", 1)[-1]
        row["cells"][cell] = f"{r['score_openjev']}-{r['score_opponent']}"

    fallback = sum((r.get("openjev") or {}).get("fallback") or 0 for r in results)
    decisions = sum((r.get("openjev") or {}).get("decisions") or 0 for r in results)
    wall_h = sum(r.get("wall_s") or 0 for r in results) / 3600

    print(f"{args.run_dir}: {len(results)} matches, {wall_h:.1f} h of match wall time")
    print(f"fallback decisions: {fallback} / {decisions}" + ("   <-- NOT all OpenJev!" if fallback else ""))
    print(f"\n{'opponent':26s}  W  D  L  GF-GA  pts  " + "  ".join(f"{c:>4s}" for c in CELL_ORDER) + "  missing")
    total = {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0}
    for name, v in sorted(table.items(), key=lambda kv: -(3 * kv[1]["W"] + kv[1]["D"])):
        for k in total:
            total[k] += v[k]
        cells = "  ".join(f"{v['cells'].get(c, '·'):>4s}" for c in CELL_ORDER)
        missing = args.expect - (v["W"] + v["D"] + v["L"])
        pts = 3 * v["W"] + v["D"]
        print(
            f"{name:26s} {v['W']:2d} {v['D']:2d} {v['L']:2d}  {v['GF']:2d}-{v['GA']:<2d} {pts:4d}  {cells}"
            f"  {missing if missing > 0 else ''}"
        )
    pts = 3 * total["W"] + total["D"]
    print(
        f"{'TOTAL':26s} {total['W']:2d} {total['D']:2d} {total['L']:2d}  {total['GF']:2d}-{total['GA']:<2d} {pts:4d}"
    )
    print("\ncells: R/L = OpenJev plays right/left, k = OpenJev kicks off; score is OpenJev-opponent")

    (args.run_dir / "summary_all.json").write_text(
        json.dumps({"per_opponent": table, "total": total, "fallback": fallback}, indent=1)
    )


if __name__ == "__main__":
    main()
