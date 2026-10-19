"""flat_round.py — the flat engine without a server, in parallel on the CPU: the scripted rule or the
policy network (`--weights`) against strategies, full matches, the 4 side x kickoff cells at several
decision rates (one rate is a lucky draw, not a measurement: docs/openjev.md).

  # the rule against the round-robin's pool: 12 strategies x 4 cells x 7 rates = 336 matches
  pixi run python -m tools.flat_round pool --policy scripted --out replays/flatnet/rule_pool
  # the policy network, same protocol
  pixi run python -m tools.flat_round pool --policy net --weights replays/flatnet/bc1.npz --out replays/flatnet/bc1_pool
  # training data: every question the rule answers, against every catalog strategy
  pixi run python -m tools.flat_round all --rates 0.95,1.05 --record --out replays/flatnet/data1
  # a network against a network (self-play): opponent `net:<weights>`
  pixi run python -m tools.flat_round net:replays/flatnet/bc1.npz --policy net --weights W --out D

Each match is a fresh process (no state carries over) and writes `<tag>.result.json`, with --record
also `<tag>.npz` (`flat.RECORD_FIELDS`, for `tools/flat_net_train.py`) and with --record-vs the
opponent network's `<tag>.b.npz`; a finished match is skipped when the command is run again. Prints
every match, then W-D-L and goals per opponent.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import time
import zlib
from pathlib import Path

CELLS = (("_Rk", True, True), ("_R", True, False), ("_Lk", False, True), ("_L", False, False))
RATES = (1.0, 0.98, 1.02, 0.96, 1.04, 0.94, 1.06)


def _play(job: dict) -> dict:
    from tools.evaluation.match import run_match
    from utama_core.strategy.openjev.flat import build_openjev_flat_kernel_strategy

    out = Path(job["out"])
    built: list = []
    built_b: list = []
    factory_b = None
    if job["vs_weights"]:  # a network on the other side too
        factory_b = functools.partial(
            build_openjev_flat_kernel_strategy,
            policy="net",
            weights=job["vs_weights"],
            decision_hz=job["rate"],
            record_path=out / f"{job['tag']}.b.npz" if job["record_vs"] else None,
            temperature=job["vs_temperature"],
            seed=job["seed"] + 1,
            on_build=built_b.append,
        )
    factory = functools.partial(
        build_openjev_flat_kernel_strategy,
        policy=job["policy"],
        weights=job["weights"],
        decision_hz=job["rate"],
        record_path=out / f"{job['tag']}.npz" if job["record"] else None,
        temperature=job["temperature"],
        seed=job["seed"],
        on_build=built.append,
    )
    t0 = time.time()
    try:
        r = run_match(
            "flat",
            job["opponent"],
            duration_seconds=job["duration"],
            a_is_right=job["is_right"],
            a_kicks_off=job["kicks_off"],
            factory_a=factory,
            factory_b=factory_b,
        )
    finally:
        for tactic in built + built_b:
            tactic.close()
    row = {
        "tag": job["tag"],
        "opponent": job["label"],
        "vs_weights": job["vs_weights"],
        "cell": job["cell"],
        "rate": job["rate"],
        "policy": job["policy"],
        "weights": job["weights"],
        "gf": r.score_a,
        "ga": r.score_b,
        "wall_s": round(time.time() - t0, 1),
        "flat": built[0].summary() if built else None,
        "flat_b": built_b[0].summary() if built_b else None,
    }
    (out / f"{job['tag']}.result.json").write_text(json.dumps(row, indent=1))
    return row


def table(rows: list[dict]) -> str:
    lines = [f"{'opponent':26s}  W  D  L   goals"]
    tot = [0, 0, 0, 0, 0]
    for opp in sorted({r["opponent"] for r in rows}):
        mine = [r for r in rows if r["opponent"] == opp]
        w = sum(r["gf"] > r["ga"] for r in mine)
        d = sum(r["gf"] == r["ga"] for r in mine)
        gf, ga = sum(r["gf"] for r in mine), sum(r["ga"] for r in mine)
        lines.append(f"{opp:26s} {w:2d} {d:2d} {len(mine) - w - d:2d}   {gf}-{ga}")
        for k, v in enumerate((w, d, len(mine) - w - d, gf, ga)):
            tot[k] += v
    n = max(1, len(rows))
    lines.append(
        f"{'TOTAL':26s} {tot[0]:2d} {tot[1]:2d} {tot[2]:2d}   {tot[3]}-{tot[4]}  {(3 * tot[0] + tot[1]) / n:.2f} pts/match"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "opponents", nargs="*", default=["pool"], help="short names, 'pool' (default), 'all' or net:<weights>"
    )
    ap.add_argument("--policy", choices=("scripted", "net"), default="scripted")
    ap.add_argument("--weights", default=None, help="the network's weights (--policy net)")
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="the network's: 0 plays its best action, above 0 draws one"
    )
    ap.add_argument("--seed", type=int, default=0, help="the draws' seed (with each match's tag)")
    ap.add_argument("--cells", default="Rk,R,Lk,L")
    ap.add_argument("--rates", default=",".join(f"{r:g}" for r in RATES), help="decisions per sim second")
    ap.add_argument("--duration", type=float, default=1200.0, help="sim-time cap; a match ends at full time")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--record", action="store_true", help="save every question (training data)")
    ap.add_argument("--vs-temperature", type=float, default=0.0, help="a net:<weights> opponent's temperature")
    ap.add_argument("--record-vs", action="store_true", help="save a net:<weights> opponent's questions too")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.policy == "net" and not a.weights:
        ap.error("--policy net needs --weights")

    from jev_tournament import resolve_opponents
    from tools.evaluation.match import _short_name

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = [c for c in CELLS if c[0][1:] in a.cells.split(",")]
    rates = [float(r) for r in a.rates.split(",")]
    names = [o for o in a.opponents if not o.startswith("net:")]
    opponents = [(name, _short_name(name), None) for name in (resolve_opponents(names) if names else [])]
    opponents += [
        ("flat_net", f"vs-{Path(o[4:]).stem}", str(Path(o[4:]).resolve())) for o in a.opponents if o.startswith("net:")
    ]
    jobs, done = [], []
    for opponent, label, vs_weights in opponents:
        for suffix, is_right, kicks_off in cells:
            for rate in rates:
                tag = f"{label}{suffix}_{rate:g}"
                if (out / f"{tag}.result.json").exists():
                    done.append(json.loads((out / f"{tag}.result.json").read_text()))
                    continue
                jobs.append(
                    dict(
                        tag=tag,
                        opponent=opponent,
                        label=label,
                        vs_weights=vs_weights,
                        vs_temperature=a.vs_temperature,
                        record_vs=a.record_vs,
                        cell=suffix[1:],
                        is_right=is_right,
                        kicks_off=kicks_off,
                        rate=rate,
                        policy=a.policy,
                        weights=str(Path(a.weights).resolve()) if a.weights else None,
                        record=a.record,
                        temperature=a.temperature,
                        seed=zlib.crc32(f"{a.seed}:{tag}".encode()),
                        duration=a.duration,
                        out=str(out),
                    )
                )
    print(f"{len(jobs)} matches to play, {len(done)} done, {a.workers} workers -> {out}", flush=True)
    rows = list(done)
    if jobs:
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:
            for k, r in enumerate(pool.imap_unordered(_play, jobs), 1):
                rows.append(r)
                f = r["flat"] or {}
                agree = f" agree {f['agree_with_rule'] / max(1, f['questions']):.3f}" if "agree_with_rule" in f else ""
                print(
                    f"[{k}/{len(jobs)}] {r['tag']:40s} {r['gf']:2d}-{r['ga']:<2d} decisions {f.get('decisions')}"
                    f"{agree} ({r['wall_s']:.0f}s)",
                    flush=True,
                )
    print()
    print(table(rows))
    (out / "summary.txt").write_text(table(rows) + "\n")


if __name__ == "__main__":
    main()
