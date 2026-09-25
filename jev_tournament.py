"""jev_tournament.py — OpenJev (via a running FOR-Engine server) vs kernel strategies, offline in rsim.

OpenJev always plays yellow (tournament_lib's `config_a` convention); side and
kickoff are swept as independent cells. Matches run sequentially — the MLX
backend serves one request at a time — and each finished match writes its own
`<tag>.result.json`, so re-running the same command with the same `--out-dir`
resumes where it stopped.

Per match, under the run directory:
  <tag>.result.json        score, match stats, OpenJev decision counters
  <tag>.openjev.jsonl      one row per OpenJev decision (state, probabilities, choice, latency…)
  <tag>.intentions.jsonl   kernel tactic-assignment log (utama_core.engine.match_log)
  <tag>.stats.json         possession / shots / ball travel
  replays/<run>/<tag>.npz  columnar replay (render with `python -m tools.jev_replay_html`)

Usage:
  # one match, watch it live
  pixi run python jev_tournament.py tiki_taka --duration 30 --decision-hz 10 --render
  # every catalog strategy, both sides, resumable
  pixi run python jev_tournament.py all --duration 65 --cells 2 --out-dir replays/jev_all
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

from tournament_lib import _CONFIG_NAMES, _short_name, run_match
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.strategy.openjev import (
    OpenJevClient,
    OpenJevPartitioner,
    build_openjev_kernel_strategy,
)

# (tag suffix, openjev_is_right, openjev_kicks_off)
CELLS = {
    1: [("_Rk", True, True)],
    2: [("_Rk", True, True), ("_L", False, False)],
    4: [
        ("_Rk", True, True),
        ("_R", True, False),
        ("_Lk", False, True),
        ("_L", False, False),
    ],
}


def resolve_opponents(names: list[str]) -> list[str]:
    if names == ["all"]:
        return sorted(_CONFIG_NAMES)
    full = [
        n if n.startswith("build_") else f"build_{n}_kernel_strategy" for n in names
    ]
    unknown = [n for n in full if n not in _CONFIG_NAMES]
    if unknown:
        raise SystemExit(
            f"unknown strategies {unknown}; available: {sorted(_short_name(n) for n in _CONFIG_NAMES)}"
        )
    return full


def play(
    opponent: str,
    cell: tuple[str, bool, bool],
    run_dir: Path,
    *,
    url: str,
    duration: float,
    decision_hz: float,
    render: bool,
) -> dict:
    suffix, is_right, kicks_off = cell
    tag = f"openjev_vs_{_short_name(opponent)}{suffix}"
    built: list[OpenJevPartitioner] = []
    factory = functools.partial(
        build_openjev_kernel_strategy,
        url=url,
        decision_hz=decision_hz,
        log_path=run_dir / f"{tag}.openjev.jsonl",
        on_build=built.append,
    )
    t0 = time.time()
    try:
        result = run_match(
            "openjev",
            opponent,
            duration_seconds=duration,
            a_is_right=is_right,
            a_kicks_off=kicks_off,
            run_dir=run_dir,
            match_tag_suffix=suffix,
            factory_a=factory,
            render=render,
        )
    finally:
        for p in built:
            p.close()
    out = {
        "tag": tag,
        "opponent": _short_name(opponent),
        "openjev_is_right": is_right,
        "openjev_kicks_off": kicks_off,
        "score_openjev": result.score_a,
        "score_opponent": result.score_b,
        "outcome": (
            "win"
            if result.score_a > result.score_b
            else "loss" if result.score_a < result.score_b else "draw"
        ),
        "duration_s": duration,
        "decision_hz": decision_hz,
        "wall_s": round(time.time() - t0, 1),
        "openjev": built[0].summary() if built else None,
        "stats": result.stats,
    }
    (run_dir / f"{tag}.result.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OpenJev vs kernel strategies (rsim, offline)"
    )
    ap.add_argument(
        "opponents",
        nargs="+",
        help="short strategy names (e.g. tiki_taka counter_flow) or 'all'",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=65.0,
        help="sim seconds per match (default 65)",
    )
    ap.add_argument(
        "--decision-hz",
        type=float,
        default=60.0,
        help="OpenJev decisions per sim second (60 = every tick)",
    )
    ap.add_argument(
        "--cells",
        type=int,
        choices=sorted(CELLS),
        default=1,
        help="side x kickoff cells per opponent",
    )
    ap.add_argument("--url", default="http://127.0.0.1:3000", help="FOR-Engine server")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="run directory (default replays/jev_<timestamp>); reuse to resume",
    )
    ap.add_argument("--render", action="store_true", help="open the live rsim window")
    args = ap.parse_args()

    opponents = resolve_opponents(args.opponents)
    try:
        version = OpenJevClient(args.url).version()
    except Exception as e:
        raise SystemExit(f"FOR-Engine not reachable at {args.url}: {e}")

    run_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPLAY_BASE_PATH / f"jev_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({"args": vars(args), "server": version}, indent=1)
    )
    print(
        f"run dir: {run_dir}\nserver: {version.get('model_dir')} ({version.get('backend', {}).get('backend')})"
    )

    results = []
    for opponent in opponents:
        for cell in CELLS[args.cells]:
            tag = f"openjev_vs_{_short_name(opponent)}{cell[0]}"
            done = run_dir / f"{tag}.result.json"
            if done.exists():
                r = json.loads(done.read_text())
                print(
                    f"  {tag:45s} (done) {r['score_openjev']}-{r['score_opponent']} {r['outcome']}"
                )
            else:
                r = play(
                    opponent,
                    cell,
                    run_dir,
                    url=args.url,
                    duration=args.duration,
                    decision_hz=args.decision_hz,
                    render=args.render,
                )
                oj = r["openjev"] or {}
                print(
                    f"  {tag:45s} {r['score_openjev']}-{r['score_opponent']} {r['outcome']:5s} "
                    f"decisions={oj.get('decisions')} cached={oj.get('cached')} fallback={oj.get('fallback')} "
                    f"mean={oj.get('mean_latency_ms')}ms wall={r['wall_s']}s"
                )
            results.append(r)

    table: dict[str, dict[str, int]] = {}
    for r in results:
        row = table.setdefault(
            r["opponent"], {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0}
        )
        row[{"win": "W", "draw": "D", "loss": "L"}[r["outcome"]]] += 1
        row["GF"] += r["score_openjev"]
        row["GA"] += r["score_opponent"]
    total = {k: sum(v[k] for v in table.values()) for k in ("W", "D", "L", "GF", "GA")}
    (run_dir / "summary.json").write_text(
        json.dumps({"per_opponent": table, "total": total}, indent=1)
    )
    print("\nOpenJev vs            W  D  L   GF-GA")
    for name, v in sorted(table.items()):
        print(f"  {name:20s} {v['W']:2d} {v['D']:2d} {v['L']:2d}   {v['GF']}-{v['GA']}")
    print(
        f"  {'TOTAL':20s} {total['W']:2d} {total['D']:2d} {total['L']:2d}   {total['GF']}-{total['GA']}"
    )


if __name__ == "__main__":
    main()
