"""jev_tournament.py — OpenJev (via a running FOR-Engine server) vs kernel strategies, offline in rsim.

OpenJev always plays yellow (`tools/evaluation/match.py`'s `config_a` convention); side and
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

`--engine split` (default): OpenJev picks a tactic split (`OpenJevPartitioner`). `--engine flat`:
OpenJev picks every outfield robot's action (`strategy/openjev/flat.py`), each question carrying the
coaching's call; `flat-nocall`: the same without the call; `flat-scripted`: the flat menu played by its
fixed rule, no server. Tags start `openjev_`, `jevflat_`, `jevnocall_`, `flatscripted_`.

Usage:
  # one match, watch it live
  pixi run python jev_tournament.py tiki_taka --duration 30 --decision-hz 10 --render
  # every catalog strategy, both sides, resumable
  pixi run python jev_tournament.py all --duration 65 --cells 2 --out-dir replays/jev_all
  # flat engine, full matches against the round-robin's pool (or tools/jev_flat_round.sh)
  pixi run python jev_tournament.py pool --engine flat --duration 1200 --cells 4 --out-dir replays/jevflat_pool
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

from tools.evaluation.match import _CONFIG_NAMES, _short_name, run_match
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.strategy.openjev import OpenJevClient, build_openjev_kernel_strategy
from utama_core.strategy.openjev.flat import build_openjev_flat_kernel_strategy

# --engine -> (tag prefix, needs a server, factory(**common) -> build_kernel_strategy factory)
ENGINES = {
    "split": ("openjev", True, build_openjev_kernel_strategy),
    "flat": ("jevflat", True, build_openjev_flat_kernel_strategy),
    "flat-nocall": ("jevnocall", True, functools.partial(build_openjev_flat_kernel_strategy, call=False)),
    "flat-scripted": (
        "flatscripted",
        False,
        functools.partial(build_openjev_flat_kernel_strategy, policy="scripted"),
    ),
}

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

# A match with more of its questions answered by the scripted fallback than this is not a model result:
# the server went down mid-match, and every question after that played the rule.
MAX_FALLBACK_SHARE = 0.01


def resolve_opponents(names: list[str]) -> list[str]:
    if names == ["all"]:
        return sorted(_CONFIG_NAMES)
    if names == ["pool"]:  # the round-robin's pool: every config not retired
        from tools.evaluation.round_robin import RETIRED

        return sorted(n for n in _CONFIG_NAMES if n not in RETIRED)
    full = [n if n.startswith("build_") else f"build_{n}_kernel_strategy" for n in names]
    unknown = [n for n in full if n not in _CONFIG_NAMES]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; available: {sorted(_short_name(n) for n in _CONFIG_NAMES)}")
    return full


def _live_note(built: list) -> str:
    if not built:
        return ""
    s = built[0].summary()
    return f"dec={s['decisions']} fb={s['fallback']} {s['mean_latency_ms']}ms"


def play(
    opponent: str,
    cell: tuple[str, bool, bool],
    run_dir: Path,
    *,
    engine: str,
    url: str,
    duration: float,
    decision_hz: float,
    render: bool,
) -> dict:
    suffix, is_right, kicks_off = cell
    prefix, _, build = ENGINES[engine]
    tag = f"{prefix}_vs_{_short_name(opponent)}{suffix}"
    built: list = []
    factory = functools.partial(
        build,
        url=url,
        decision_hz=decision_hz,
        log_path=run_dir / f"{tag}.openjev.jsonl",
        on_build=built.append,
    )
    t0 = time.time()
    try:
        result = run_match(
            prefix,
            opponent,
            duration_seconds=duration,
            a_is_right=is_right,
            a_kicks_off=kicks_off,
            run_dir=run_dir,
            match_tag_suffix=suffix,
            factory_a=factory,
            render=render,
            progress=True,
            progress_note=lambda: _live_note(built),
        )
    finally:
        for p in built:
            p.close()
    out = {
        "tag": tag,
        "engine": engine,
        "opponent": _short_name(opponent),
        "openjev_is_right": is_right,
        "openjev_kicks_off": kicks_off,
        "score_openjev": result.score_a,
        "score_opponent": result.score_b,
        "outcome": (
            "win" if result.score_a > result.score_b else "loss" if result.score_a < result.score_b else "draw"
        ),
        "duration_s": duration,
        "decision_hz": decision_hz,
        "wall_s": round(time.time() - t0, 1),
        "openjev": built[0].summary() if built else None,
        "stats": result.stats,
    }
    oj = out["openjev"] or {}
    if oj.get("questions") and oj["fallback"] > MAX_FALLBACK_SHARE * oj["questions"]:
        # Kept apart from the results, and the run stops: the matches after it would fall back too.
        (run_dir / f"{tag}.invalid.json").write_text(json.dumps(out, indent=1, default=str))
        raise SystemExit(
            f"{tag}: {oj['fallback']} of {oj['questions']} questions fell back to the scripted rule "
            "(is the server down?); not recorded, re-run the same command once it is back"
        )
    (run_dir / f"{tag}.result.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenJev vs kernel strategies (rsim, offline)")
    ap.add_argument(
        "opponents",
        nargs="+",
        help="short strategy names (e.g. tiki_taka counter_flow), 'all', or 'pool' (the round-robin's)",
    )
    ap.add_argument(
        "--engine",
        choices=sorted(ENGINES),
        default="split",
        help="split: OpenJev picks a tactic split; flat: OpenJev picks every robot's action, advised by the "
        "coaching's call; flat-nocall: without the call; flat-scripted: flat's menu played by a fixed rule, no server",
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
        default=None,
        help="OpenJev decisions per sim second (60 = every tick; default 60 for split, 1 for flat, which also "
        "decides at once when the situation changes: carrier, possession, ball cell, a kick)",
    )
    ap.add_argument(
        "--cells",
        type=int,
        choices=sorted(CELLS),
        default=1,
        help="side x kickoff cells per opponent",
    )
    ap.add_argument(
        "--cell-tags",
        default=None,
        help="only these cells of --cells 4, e.g. Rk,L (for splitting one opponent across workers)",
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
    prefix, needs_server, _ = ENGINES[args.engine]
    if args.decision_hz is None:
        args.decision_hz = 60.0 if args.engine == "split" else 1.0
    version: dict = {}
    if needs_server:
        try:
            version = OpenJevClient(args.url).version()
        except Exception as e:
            raise SystemExit(f"FOR-Engine not reachable at {args.url}: {e}")

    run_dir = Path(args.out_dir) if args.out_dir else REPLAY_BASE_PATH / f"jev_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"args": vars(args), "server": version}, indent=1))
    print(f"run dir: {run_dir}\nserver: {version.get('model_dir')} ({version.get('backend', {}).get('backend')})")

    cells = CELLS[args.cells]
    if args.cell_tags:
        wanted = {f"_{t}" for t in args.cell_tags.split(",")}
        cells = [c for c in CELLS[4] if c[0] in wanted]
    results = []
    n_total = len(opponents) * len(cells)
    for opponent in opponents:
        for cell in cells:
            tag = f"{prefix}_vs_{_short_name(opponent)}{cell[0]}"
            print(f"[{len(results) + 1}/{n_total}] {tag}")
            done = run_dir / f"{tag}.result.json"
            if done.exists():
                r = json.loads(done.read_text())
                print(f"  {tag:45s} (done) {r['score_openjev']}-{r['score_opponent']} {r['outcome']}")
            else:
                r = play(
                    opponent,
                    cell,
                    run_dir,
                    engine=args.engine,
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
        row = table.setdefault(r["opponent"], {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0})
        row[{"win": "W", "draw": "D", "loss": "L"}[r["outcome"]]] += 1
        row["GF"] += r["score_openjev"]
        row["GA"] += r["score_opponent"]
    total = {k: sum(v[k] for v in table.values()) for k in ("W", "D", "L", "GF", "GA")}
    (run_dir / "summary.json").write_text(json.dumps({"per_opponent": table, "total": total}, indent=1))
    print("\nOpenJev vs            W  D  L   GF-GA")
    for name, v in sorted(table.items()):
        print(f"  {name:20s} {v['W']:2d} {v['D']:2d} {v['L']:2d}   {v['GF']}-{v['GA']}")
    print(f"  {'TOTAL':20s} {total['W']:2d} {total['D']:2d} {total['L']:2d}   {total['GF']}-{total['GA']}")


if __name__ == "__main__":
    main()
