"""evaluate.py — is a candidate strategy better than the champion? As few matches as that takes.

Run:
    pixi run python tools/evaluation/evaluate.py CANDIDATE [CANDIDATE ...] [flags]

    --pool PATH       the reference pool (default tools/evaluation/pools/pool_v1.json): the
                      champion, the opponents, the held-out opponents and the sim noise
    --held-out        play the pool's held-out opponents instead (the check before promoting)
    --baseline NAME   compare against this strategy instead of the champion: a new version
                      against the one it changes. The baseline takes the champion's role, the
                      champion joins the opponents, and a baseline from the pool leaves them
    --max-rounds N    rounds before a candidate is called undecided (default 6)
    --max-workers N   matches at once (default 8)
    --explain         also write on-pitch pictures of each candidate against the champion to
                      replays/eval/explain/<candidate>_<run id>/ (positions, shots, fouls and a
                      summary; see tools/evaluation/explain.py). Explains the verdict, never
                      changes it. Off by default; the matches are kept until it has run.

How it decides
--------------
Matches are played in rounds. In round k each candidate plays every opponent and the champion
once, and the champion plays every opponent, all with sim-noise seed k, so the candidate and the
champion meet the same opponents under the same noise. The candidate is config_a in odd rounds
and config_b in even ones.

After each round the candidate is compared with the champion on paired matches
(`ratings.paired_difference`): its goal difference minus the champion's against the same
opponent with the same seed and colour, from every match in the ledger (`replay/eval_ledger.py`:
this run's, earlier runs', round-robins'), plus their head-to-head matches. Paired results share
the opponent and the noise, so their difference cancels much of it: all of it for identical
code, little for unrelated strategies; the spread is estimated from the differences themselves.
P(candidate stronger than champion) decides:

- **better** once P >= PROMOTE (0.99),
- **worse** once P <= DROP (0.2),
- **undecided** after `--max-rounds`: not clearly better, which for a search means drop it.

Thresholds come from simulating the rule (match noise SD 1.9 goals, six opponents, 6 rounds,
paired results correlated 0 to 1): two equal strategies are called "better" at most 3% of the
time. Unrelated strategies (correlation 0): one goal per match worse is dropped after ~2 rounds,
one goal better found 61% of the time, half a goal better 20%. A small change to a strategy
(correlation ~0.9): one goal better found every time in ~3 rounds, half a goal better 55%.
Strengths printed at the end come from the ratings over all matches (`analysis/ratings.py`).

A fixture already in the ledger is never played again, so the champion's matches are played
once per seed and reused by every candidate. Replays are not kept: a match is a pure function
of its fixture, so `round_robin.py --pair A B --sim-noise` with the pool's noise and the seed
plays it again.

Promoting is a decision, not a side effect: a candidate called better should pass `--held-out`
(it must not lose clearly to opponents never used for selection) before you set it as the
pool's champion by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import _short_name, run_match  # noqa: E402
from tools.evaluation.round_robin import (  # noqa: E402
    MAX_MATCH_SECONDS,
    _resolve_config_names,
    _run_metadata,
)
from utama_core.analysis import ratings  # noqa: E402
from utama_core.config.settings import REPLAY_BASE_PATH  # noqa: E402
from utama_core.replay import eval_ledger  # noqa: E402
from utama_core.replay.fingerprint import CodeGraph  # noqa: E402
from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise  # noqa: E402

DEFAULT_POOL = Path(__file__).with_name("pools") / "pool_v1.json"
PROMOTE = 0.99
DROP = 0.2
MAX_ROUNDS = 6


def round_fixtures(candidates: list[str], champion: str, opponents: list[str], k: int) -> list[tuple[str, str, int]]:
    """Round `k`'s (config_a, config_b, seed): each candidate against every opponent and the
    champion, and the champion against every opponent; the side under test is config_a in odd
    rounds."""
    out = []
    for side in [*candidates, champion]:
        rivals = opponents if side == champion else [*opponents, champion]
        for o in rivals:
            out.append((side, o, k) if k % 2 else (o, side, k))
    return list(dict.fromkeys(out))


def verdict(p_better: float, rounds: int, max_rounds: int) -> str | None:
    """The candidate's verdict after `rounds` rounds, or None to play another."""
    if p_better >= PROMOTE:
        return "better"
    if p_better <= DROP:
        return "worse"
    return "undecided" if rounds >= max_rounds else None


def play(a: str, b: str, seed: int, noise: dict, run_dir: Path, keep: bool = False) -> dict:
    """One fixture's `MatchResult` as a dict. The match is recorded like a round-robin's (same
    play), then its files are deleted unless `keep` (`--explain` reads them): only the result is kept."""
    tag = f"{_short_name(a)}_vs_{_short_name(b)}_s{seed}"
    result = run_match(
        a,
        b,
        duration_seconds=MAX_MATCH_SECONDS,
        run_dir=run_dir,
        match_tag_suffix=f"_s{seed}",
        sim_noise=SimNoise.from_dict({**noise, "seed": seed}),
    )
    if not keep:
        for f in run_dir.glob(f"{tag}.*"):
            f.unlink()
    return {
        "config_a": result.config_a,
        "config_b": result.config_b,
        "score_a": result.score_a,
        "score_b": result.score_b,
        "stats": {"shots": (result.stats or {}).get("shots")},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--max-workers", type=int, default=min(8, max(1, (os.cpu_count() or 1) - 1)))
    parser.add_argument("--explain", action="store_true", help="also write on-pitch pictures per candidate (see above)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pool = json.loads(args.pool.read_text())
    champion = _resolve_config_names([args.baseline or pool["champion"]])[0]
    opponents = _resolve_config_names(pool["held_out"] if args.held_out else pool["opponents"])
    pool_champion = _resolve_config_names([pool["champion"]])[0]
    if champion != pool_champion:
        opponents.append(pool_champion)  # against a baseline, the champion is one more opponent
    opponents = [o for o in opponents if o != champion]  # a baseline from the pool leaves its seat
    candidates = [c for c in _resolve_config_names(args.candidates) if c != champion]
    noise = SimNoise.from_dict(pool["sim_noise"]).to_dict()
    noise.pop("seed", None)

    graph = CodeGraph()
    env = eval_ledger.environment(graph, duration_seconds=MAX_MATCH_SECONDS, sim_noise=noise)
    fps = {n: graph.strategy_fingerprint(n)[:16] for n in {*candidates, champion, *opponents}}

    def fixture_id(f: tuple[str, str, int]) -> str:
        a, b, seed = f
        return eval_ledger.fixture_id(env, a, fps[a], b, fps[b], seed)

    run_id = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
    run_dir = Path(REPLAY_BASE_PATH) / run_id  # replay files must sit directly under replays/
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Pool v{pool['version']}{' (held out)' if args.held_out else ''}: "
        f"{'baseline' if args.baseline else 'champion'} {_short_name(champion)}; "
        f"opponents {', '.join(_short_name(o) for o in opponents)}; environment {env}\n"
    )

    open_ = list(candidates)
    report: dict[str, dict] = {}
    played = reused = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for k in range(1, args.max_rounds + 1):
            if not open_:
                break
            ledger = eval_ledger.load(env)
            todo = [f for f in round_fixtures(open_, champion, opponents, k) if fixture_id(f) not in ledger]
            reused += len(round_fixtures(open_, champion, opponents, k)) - len(todo)
            futures = {executor.submit(play, *f, noise, run_dir, args.explain): f for f in todo}
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    a, b, seed = futures.pop(future)
                    result = future.result()
                    eval_ledger.append([eval_ledger.entry(env, graph, result, seed, source=run_id)])
                    played += 1
                    print(
                        f"  round {k}: {_short_name(a):<24} {result['score_a']} - {result['score_b']} {_short_name(b)}",
                        flush=True,
                    )

            fit = _fit(env, fps, {*open_, champion, *opponents})
            ledger = _up_to_round(eval_ledger.load(env), k)
            for c in list(open_):
                margin, sd = ratings.paired_difference(*paired(ledger, fps, c, champion, opponents), fit.noise_sd)
                p = ratings.p_positive(margin, sd)
                v = verdict(p, k, args.max_rounds)
                print(f"after round {k}: {_short_name(c):<28} P(better) {p:.2f}  margin {margin:+.2f} ± {sd:.2f}")
                if v is not None:
                    open_.remove(c)
                    report[c] = {"verdict": v, "rounds": k, "p_better": round(p, 3), "margin": round(margin, 2)}
                    report[c]["margin_sd"] = round(sd, 2)
            print()
    if args.explain:
        _explain(run_dir, run_id, candidates, champion)
    shutil.rmtree(run_dir, ignore_errors=True)  # this run's own scratch dir; results are in the ledger

    final = _fit(env, fps, {*candidates, champion, *opponents})
    ledger = eval_ledger.load(env)
    print(f"Verdicts ({played} matches played, {reused} reused from the ledger):")
    for c in candidates:
        r = report[c]
        print(
            f"  {_short_name(c):<28} {r['verdict'].upper():<10} P(better) {r['p_better']:.2f}  "
            f"margin {r['margin']:+.2f} ± {r['margin_sd']:.2f} goals/match  after {r['rounds']} round(s)"
        )
        r["per_opponent"] = _per_opponent(_up_to_round(ledger, r["rounds"]), fps, c, champion, opponents)
        print(f"    {'opponent':<26} {'candidate GD':>12} {'champion GD':>12} {'matches':>8} (same seed and colour)")
        for o, row in r["per_opponent"].items():
            name = o if o == "head-to-head" else _short_name(o)
            print(f"    {name:<26} {row['candidate']:>+12} {row['champion']:>+12} {row['matches']:>8}")
        if r["verdict"] == "better" and not args.held_out:
            print(f"    -> next: evaluate.py {_short_name(c)} --held-out, then set it as the pool's champion")
    print("\nStrengths (goals per match, ± SD):")
    for n in sorted(final.names, key=lambda n: -final.strength(n)[0]):
        m, sd = final.strength(n)
        print(f"  {_short_name(n):<28} {m:+.2f} ± {sd:.2f}")

    out = Path(REPLAY_BASE_PATH) / "eval" / "reports" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run": _run_metadata(),
                "pool": str(args.pool),
                "held_out": args.held_out,
                "environment": env,
                "rule": {"promote": PROMOTE, "drop": DROP, "max_rounds": args.max_rounds},
                "candidates": report,
                "strengths": {n: final.strength(n) for n in final.names},
                "noise_sd": final.noise_sd,
            },
            indent=2,
        )
    )
    print(f"\nReport: {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")


def _explain(run_dir: Path, run_id: str, candidates: list[str], champion: str) -> None:
    """`--explain`: pictures of each candidate against the champion, from this run's kept matches.
    Runs before the scratch dir is removed; a failure is printed and never loses the verdicts."""
    from tools.evaluation import explain  # matplotlib: imported only when asked for

    for c in candidates:
        out = Path(REPLAY_BASE_PATH) / "eval" / "explain" / f"{_short_name(c)}_{run_id}"
        try:
            explain.explain(run_dir, c, champion, out)
            print(f"explain: {_short_name(c)} vs {_short_name(champion)} written to {out}")
        except Exception as exc:  # noqa: BLE001 - the pictures are optional; the verdicts are not
            print(f"explain: {_short_name(c)} failed ({type(exc).__name__}: {exc}); verdicts are unaffected")


def _fit(env: str, fps: dict[str, str], players: set[str]) -> ratings.Ratings:
    """Ratings over every ledger match in `env` between `players` at their current code."""
    matches = [
        (e["a"], e["b"], e["score_a"] - e["score_b"])
        for e in eval_ledger.load(env).values()
        if e["a"] in players and e["b"] in players and fps.get(e["a"]) == e["a_fp"] and fps.get(e["b"]) == e["b_fp"]
    ]
    return ratings.fit(matches, names=sorted(players))


def _up_to_round(ledger: dict, k: int) -> dict:
    """The ledger's matches of rounds 1..k (seeds 1..k): a verdict after round k uses those only,
    whatever later rounds the ledger already holds, so it doesn't depend on earlier runs."""
    return {f: e for f, e in ledger.items() if e["seed"] is not None and e["seed"] <= k}


def _margins(ledger: dict, fps: dict, side: str, o: str) -> dict[tuple, int]:
    """{(seed, side is config_a): goal difference} for `side` against `o`, at current code."""
    out = {}
    for e in ledger.values():
        if e["a_fp"] != fps.get(e["a"]) or e["b_fp"] != fps.get(e["b"]):
            continue
        if (e["a"], e["b"]) == (side, o):
            out[(e["seed"], True)] = e["score_a"] - e["score_b"]
        elif (e["a"], e["b"]) == (o, side):
            out[(e["seed"], False)] = e["score_b"] - e["score_a"]
    return out


def paired(ledger: dict, fps: dict, candidate: str, champion: str, opponents: list[str]) -> tuple[list, list]:
    """The candidate's goal difference minus the champion's against each opponent, over the
    matches both played with the same seed and colour, and the candidate's margins against the
    champion: `ratings.paired_difference`'s inputs."""
    diffs = []
    for o in opponents:
        c, ch = _margins(ledger, fps, candidate, o), _margins(ledger, fps, champion, o)
        diffs += [c[k] - ch[k] for k in sorted(set(c) & set(ch))]
    return diffs, list(_margins(ledger, fps, candidate, champion).values())


def _per_opponent(ledger: dict, fps: dict, candidate: str, champion: str, opponents: list[str]) -> dict:
    """Goal difference against each opponent, the candidate's and the champion's, over the
    matches both played with the same seed and colour (so the two columns face the same noise)."""
    rows = {}
    for o in opponents:
        c, ch = _margins(ledger, fps, candidate, o), _margins(ledger, fps, champion, o)
        common = sorted(set(c) & set(ch))
        rows[o] = {
            "candidate": sum(c[k] for k in common),
            "champion": sum(ch[k] for k in common),
            "matches": len(common),
        }
    h2h = _margins(ledger, fps, candidate, champion)
    rows["head-to-head"] = {"candidate": sum(h2h.values()), "champion": -sum(h2h.values()), "matches": len(h2h)}
    return rows


if __name__ == "__main__":
    main()
