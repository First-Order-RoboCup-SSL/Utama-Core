"""evaluate.py — how the strategies play, and whether a candidate beats the champion at the last release.

Run:
    pixi run python tools/evaluation/evaluate.py [CANDIDATE ...] [flags]

With no candidate it plays one round: a round robin of the players, the health check (stalls, fouls,
every signal). With candidates (strategy names, played from the working tree, or `.strategy` files)
it plays rounds until each candidate is decided.

    --at REF            the reference is the champion as it was at git commit REF (default: the latest
                        release tag), played from a `.strategy` file built once into replays/eval/frozen/
    --all               every strategy in the catalog plays, not only the core (occasionally, for a
                        wider field)

Matches play in parallel, one fewer than the CPUs (at most 12). To explain a result: find_bugs.py,
goal_sources, pitch_maps.py and explain.py on the run's round folders.

Players
-------
`reference/players.json` names the champion and the core strategies, which play every round. Every
other strategy in the catalog is reserve: it plays only with `--all`, an occasional run for a wider
field (the verdict needs only fixed opponents, the same for candidate and reference). The reference (the champion
at the last release) plays every round too. It is a frozen file, so a change to shared code moves the
candidates and the working-tree strategies, never the reference.

Rounds
------
Round k plays under sim-noise seed k (the reference world and robots, reference/world.json and
robots.json). Every round is the same round robin: every pair of players once, under seed k. Only the
candidates still undecided stay in later rounds. Nothing is played twice: the match results store holds
every match whose code is unchanged, so a run plays only the matches of edited strategies, and of a
seed not played before. In every match the candidate (then the reference) is config_a in odd rounds and
config_b in even ones, so a candidate and the reference meet each opponent with the same colour and
noise. Matches are 1200 s of sim time at most (two 300 s halves of playing time, `MAX_MATCH_SECONDS`).

How it decides
--------------
One number per opponent and seed: the candidate's goal difference minus the reference's, against the
same opponent with the same seed and colour (a "diff", `paired`). The same opponent cancels its
strength and the same noise cancels part of the luck. Head-to-head matches are reported, not counted:
they answer "does it beat the champion?", the one matchup a search is most tempted to exploit.

- **Screen**, seeds 1..ROUNDS: from round 2 on, a candidate whose mean diff is <= 0 is **dropped**.
  One that survives ROUNDS rounds goes on to confirm; the screen never accepts.
- **Confirm**, fresh seeds 101..100+ROUNDS: the same drop rule, then **better** if the one-sided
  paired t-test on these seeds' diffs gives p <= ALPHA, else **undecided** (`ratings.decide`).

Only fresh seeds can accept, so a candidate picked because it looked good on seeds 1..ROUNDS, or tuned
to them, is not accepted on that luck; dropping early can never make a false "better". "Dropped" means
not promising, not worse: an equal strategy is dropped at the first look about half the time.
`tools/evaluation/verdict_sim.py` plays this rule on simulated results (no matches); its test pins
that an equal strategy is called better at most ALPHA of the time.

ROUNDS and ALPHA (in `analysis/ratings.py`) are the rule's only settings; both are provisional. The reference's results on each
seed are stored and reused for every candidate, so a confirm stage that many candidates reach shares
the reference's luck on seeds 101..; few do, since most are dropped in the screen.

The reference is frozen with the shared code (tactics, skills, planner, engine) of the release, so a
"better" can come from shared code changed since. Today's champion plays the same opponents with the
same seeds and colours, so the report adds the candidate against it too, at no extra matches: there
the shared code is the same on both sides. `evaluate.py <champion>` measures the shared code alone.

Signals and stalls are reported, never judged on: they explain a result (docs/signals.md). The stall
watchdog watches config_a only, so the report says how many matches each strategy was checked in.

Matches the match results store holds (replays/match_results/, `replay/match_results.py`) are taken from
it, not played: after a change to one strategy only its matches play. Delete that folder to replay all.

A match that crashes is reported (`failed` in summary.json and report.json), not stored, and missing
from the verdict's pairs; the run goes on, and the next run plays it again.

Writes replays/eval_<UTC time>/: `report.json` (verdicts, round 1's signals), and per round `round_<k>/`
with each played match's replay, intention log and stats (`<a>_vs_<b>.*`), `summary.json` and
`ball_losses.md` — the layout tools/find_bugs.py, tools/pitch_maps.py and goal_sources read.

Promoting is a decision, not a side effect: set the champion by hand in reference/players.json. It is
the reference from the next release on: the reference is read from players.json as it was at the release.
"""

from __future__ import annotations

import os

# One thread per match process: several matches run at once.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import itertools
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import (  # noqa: E402
    _CONFIG_NAMES,
    TICKS_PER_SECOND,
    _short_name,
    run_match,
)
from tools.evaluation.sim_settings import load_players, reference_noise  # noqa: E402
from utama_core.analysis import (  # noqa: E402
    ratings,
    restart_outcomes,
    turnover_breakdown,
)
from utama_core.analysis.strategy_table import strategy_table  # noqa: E402
from utama_core.bundle.strategy_file import SUFFIX, frozen  # noqa: E402
from utama_core.config.settings import REPLAY_BASE_PATH  # noqa: E402
from utama_core.replay import match_results  # noqa: E402
from utama_core.replay.fingerprint import CodeGraph  # noqa: E402
from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise  # noqa: E402
from utama_core.strategy import catalog  # noqa: E402

CONFIRM_SEED = 101  # the confirm stage's first seed: never one the screen plays
WORKERS = min(12, max(1, (os.cpu_count() or 1) - 1))  # 1 plays in this process (tests)
FROZEN_DIR = Path("replays/eval/frozen")  # relative: a file's path is its name in the match results
# A full match: two halves of 300 s of playing time, the clock stopped whenever no team may play the
# ball, about 710-870 s of sim time. This only caps a match that can't reach full time.
MAX_MATCH_SECONDS = 1200.0


# --- who plays what ---------------------------------------------------------------------------


def round_fixtures(
    k: int, candidates: list[str], reference: str, opponents: list[str], seed: Optional[int] = None
) -> list[tuple[str, str, int]]:
    """Round `k`'s (config_a, config_b, seed): every pair of candidates, reference and opponents once,
    under `seed` (k by default).
    The store holds every match whose code is unchanged, so only the matches of edited strategies, or
    of a seed not played before, are actually played. The earlier of the two in (candidates,
    reference, opponents) order is config_a in odd rounds, so candidates and reference face each
    opponent with the same colour."""
    seed = k if seed is None else seed
    order = list(dict.fromkeys([*candidates, reference, *opponents]))
    pairs = itertools.combinations(order, 2)
    return [(x, y, seed) if k % 2 else (y, x, seed) for x, y in pairs]


def seed_of(k: int) -> int:
    """Round k's noise seed: k in the screen (rounds 1..ROUNDS), CONFIRM_SEED on in the confirm stage."""
    n = ratings.ROUNDS
    return k if k <= n else CONFIRM_SEED + k - n - 1


def stage_seeds(k: int) -> set[int]:
    """The seeds of round k's stage played so far: the stage's diffs, and only those, decide."""
    first = 1 if k <= ratings.ROUNDS else ratings.ROUNDS + 1
    return {seed_of(j) for j in range(first, k + 1)}


def resolve_names(requested: list[str]) -> list[str]:
    """Strategy names (today's or old ones) in catalog order, then `.strategy` files as given."""
    files = [n for n in requested if n.endswith(SUFFIX)]
    wanted = {catalog.current_name(n) for n in requested if n not in files}
    names = [n for n in _CONFIG_NAMES if n in wanted]
    if unknown := wanted - set(names):
        raise SystemExit(f"Unknown strategy name(s): {sorted(unknown)}. Available: {sorted(_CONFIG_NAMES)}")
    return names + files


def latest_release() -> str:
    """The latest release tag reachable from HEAD. Without one there is no reference: stop rather
    than quietly compare against today's code."""
    tag = _git("describe", "--tags", "--abbrev=0")
    if not tag:
        raise SystemExit("No release tag reachable from HEAD (a shallow clone?): fetch the tags or pass --at REF.")
    return tag


def champion_at(ref: str, today: str) -> str:
    """The champion that `reference/players.json` named at `ref`, or `today`'s if that commit has
    none. The reference is the champion of the last release: a strategy promoted since (perhaps
    one that didn't exist then) becomes the reference only at the next release."""
    text = _git("show", f"{ref}:tools/evaluation/reference/players.json")
    try:
        return json.loads(text)["champion"] if text else today
    except (ValueError, KeyError, TypeError):
        return today


def same_code(players: list[str], fps: dict[str, str]) -> dict[str, str]:
    """Each player whose code is the same as an earlier player's in `players`, mapped to that
    player. Right after a release the frozen champion is the working-tree champion's code: the two
    would be one player under two names, so the later one is left out."""
    first: dict[str, str] = {}
    out = {}
    for n in players:
        out.update({n: first[fps[n]]} if fps[n] in first else {})
        first.setdefault(fps[n], n)
    return out


def at_ref(name: str, ref: str) -> str:
    """Strategy `name` as it was at `ref`, as a `.strategy` file's path (built once, then kept). An
    older commit may know the strategy by its old name (`catalog.RENAMED`)."""
    for n in [name, *(old for old, new in catalog.RENAMED.items() if new == name)]:
        try:
            return frozen(n, ref, FROZEN_DIR).as_posix()
        except ValueError:
            continue
    raise SystemExit(f"no strategy {name} (or an old name of it) at {ref}")


# --- playing ----------------------------------------------------------------------------------


def analyse_replay(npz: Path) -> tuple[list[dict], dict]:
    """A played match's restart episodes and ball-loss record, from its replay. Kept out of `match.py`
    (the environment's entry point), so changing an analysis replays no match."""
    return restart_outcomes.analyse_match(npz), turnover_breakdown.analyse_match(str(npz))


def tag(a: str, b: str) -> str:
    """A match's file stem in a round directory."""
    return f"{_short_name(a)}_vs_{_short_name(b)}"


def play_and_record(
    config_a: str, config_b: str, *, a_fp: str, b_fp: str, env: str, seed: Optional[int], source: str,
    run_dir: Path, **run_kwargs,
) -> dict:  # fmt: skip
    """Play one match into `run_dir` and return its stored record (`match_results.record`), with its
    replay analysed. `run_kwargs` go to `match.run_match`."""
    result = run_match(config_a, config_b, run_dir=run_dir, **run_kwargs)
    restarts = losses = replay = None
    npz = Path(run_dir) / f"{tag(config_a, config_b)}.npz"
    if npz.exists():  # no replay: crashed before it was written
        restarts, losses = analyse_replay(npz)
        replay = Path(os.path.relpath(npz.with_suffix(""), REPLAY_BASE_PATH)).as_posix()
    return match_results.record(
        env, a_fp, b_fp, dataclasses.asdict(result), seed=seed, source=source, restarts=restarts, losses=losses,
        replay=replay,
    )  # fmt: skip


def link_replay(rec: dict, round_dir: Path, a: str, b: str) -> bool:
    """Link a reused match's files (replay, stats, intentions) into this round's folder under this
    run's names, so every reader of a round folder sees all of its matches. False if the record has
    no replay or its files were deleted."""
    if not rec.get("replay"):
        return False
    stem = Path(REPLAY_BASE_PATH) / rec["replay"]
    files = sorted(stem.parent.glob(f"{stem.name}.*"))
    if not any(f.suffix == ".npz" for f in files):
        return False
    for f in files:
        link = round_dir / f"{tag(a, b)}{f.name[len(stem.name):]}"
        if not link.exists():
            link.symlink_to(os.path.relpath(f, round_dir))
    return True


def result_row(rec: dict, a: str, b: str, reused: bool) -> dict:
    """A record as a `summary.json` result, under this run's names for the two players."""
    winner = a if rec["score_a"] > rec["score_b"] else b if rec["score_b"] > rec["score_a"] else "draw"
    return {
        "config_a": a,
        "config_b": b,
        "score_a": rec["score_a"],
        "score_b": rec["score_b"],
        "winner": winner,
        "seed": rec["seed"],
        "stats": rec["stats"],
        "possession_backstop": possession_backstop(rec["stats"]),
        "reused": reused,
    }


def possession_backstop(stats: Optional[dict]) -> bool:
    """A stall the watchdog missed: possession pinned at 100/0 with the ball barely moving."""
    if not stats:
        return False
    poss = stats.get("possession_pct") or {}
    pinned = {poss.get("friendly"), poss.get("enemy")} == {0.0, 1.0}
    return pinned and (stats.get("ball_travel_m") or 0.0) < 1.0


# --- the run ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", nargs="*")
    parser.add_argument("--at", metavar="REF")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--players", type=Path, help=argparse.SUPPRESS)  # tests: another players.json
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    setup = load_players(args.players) if args.players else load_players()
    core = resolve_names(setup["core"])
    reserve = [n for n in _CONFIG_NAMES if n not in core] if args.all else []
    ref = args.at or latest_release()
    reference = at_ref(resolve_names([champion_at(ref, setup["champion"])])[0], ref)
    today = resolve_names([setup["champion"]])[0]  # today's champion, on today's shared code
    candidates = [c for c in resolve_names(args.candidates) if c != reference]

    noise = SimNoise.from_dict(reference_noise()).to_dict()
    noise.pop("seed", None)
    env_args = {
        "duration_seconds": MAX_MATCH_SECONDS,
        "sim_noise": noise,
    }
    graph = CodeGraph()
    env = match_results.environment(graph, **env_args)
    store = match_results.open_store(REPLAY_BASE_PATH)
    everyone = list(dict.fromkeys([reference, *candidates, *core, *reserve]))
    fps = {n: graph.strategy_fingerprint(n)[:16] for n in everyone}
    for n, first in (dupes := same_code(everyone, fps)).items():
        print(f"{_short_name(n)} is left out: it is the same code as {_short_name(first)}")
    candidates, core, reserve = ([n for n in group if n not in dupes] for group in (candidates, core, reserve))
    fps = {n: fp for n, fp in fps.items() if n not in dupes}

    run_id = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
    while True:  # claim the folder: two runs (two worktrees) may start in the same second
        try:
            (Path(REPLAY_BASE_PATH) / run_id).mkdir(parents=True)
            break
        except FileExistsError:
            run_id += "b"
    run_dir = Path(REPLAY_BASE_PATH) / run_id
    print(f"Reference {_short_name(reference)}; core {', '.join(_short_name(n) for n in core)}; environment {env}")

    def still_current(a: str, b: str) -> bool:
        """Store a played match only if the code it ran is still the code on disk."""
        now = CodeGraph()
        return match_results.environment(now, **env_args) == env and all(
            now.strategy_fingerprint(n)[:16] == fps[n] for n in (a, b)
        )

    run_kwargs = {k: v for k, v in env_args.items() if k != "sim_noise"}
    open_ = list(candidates)
    report: dict[str, dict] = {}
    rounds: list[dict] = []
    opponents_so_far: set[str] = set()
    played = reused = 0
    failed: list[dict] = []
    executor = ProcessPoolExecutor(max_workers=WORKERS) if WORKERS > 1 else None
    try:
        for k in range(1, 2 * ratings.ROUNDS + 1):
            if k > 1 and not open_:
                break
            opponents = [p for p in [*core, *reserve] if p not in candidates and p != reference]
            # today's champion first: it then meets every other opponent in the candidates' colour
            opponents.sort(key=lambda p: p != today)
            opponents_so_far |= set(opponents)
            seed = seed_of(k)
            fixtures = round_fixtures(k, open_ if k > 1 else candidates, reference, opponents, seed)
            round_dir = run_dir / f"round_{k}"
            round_dir.mkdir(parents=True, exist_ok=True)
            recs: dict[tuple[str, str], tuple[dict, bool]] = {}
            todo = []
            for a, b, seed in fixtures:
                rec = store.get(env, match_results.fixture_id(env, fps[a], fps[b], seed))
                if rec is None:
                    todo.append((a, b, seed))
                else:
                    recs[(a, b)] = (rec, True)
                    link_replay(rec, round_dir, a, b)
            reused += len(recs)
            print(
                f"\nRound {k}: {len(fixtures)} matches, {len(recs)} from the match results store, {len(todo)} to play"
            )

            def finish(a: str, b: str, rec: dict) -> None:
                nonlocal played
                played += 1
                recs[(a, b)] = (rec, False)
                if still_current(a, b):
                    store.put(rec)
                else:
                    print(f"  {tag(a, b)} not stored: its code changed while the run played", flush=True)
                print(
                    f"  round {k}: {_short_name(a):<24} {rec['score_a']} - {rec['score_b']} {_short_name(b)}",
                    flush=True,
                )

            def fail(a: str, b: str, err: BaseException) -> None:
                """A match that crashed: reported, not stored, missing from the verdict's pairs."""
                print(f"  round {k}: {tag(a, b)} CRASHED: {err!r}", flush=True)
                failed.append({"round": k, "config_a": a, "config_b": b, "error": repr(err)})

            def submit(a: str, b: str, seed: int) -> tuple:
                kw = dict(a_fp=fps[a], b_fp=fps[b], env=env, seed=seed, source=run_id, run_dir=round_dir)
                kw.update(run_kwargs, sim_noise=SimNoise.from_dict({**noise, "seed": seed}))
                return a, b, kw

            if executor is None:
                for a, b, kw in (submit(*f) for f in todo):
                    try:
                        rec = play_and_record(a, b, **kw)
                    except Exception as err:  # noqa: BLE001 -- one bad match must not lose the run
                        fail(a, b, err)
                    else:
                        finish(a, b, rec)
            else:
                futures = {
                    executor.submit(play_and_record, a, b, **kw): (a, b) for a, b, kw in (submit(*f) for f in todo)
                }
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        a, b = futures.pop(future)
                        try:
                            rec = future.result()
                        except Exception as err:  # noqa: BLE001 -- one bad match must not lose the run
                            fail(a, b, err)
                        else:
                            finish(a, b, rec)

            rows = [result_row(rec, a, b, was_reused) for (a, b), (rec, was_reused) in recs.items()]
            names = sorted({n for pair in recs for n in pair})
            summary = {"run_id": f"{run_id}/round_{k}", "round": k, "seed": seed, "config_names": names}
            summary["results"] = sorted(rows, key=lambda r: (r["config_a"], r["config_b"]))
            summary["failed"] = [f for f in failed if f["round"] == k]
            summary.update(signals(summary, {pair: rec for pair, (rec, _) in recs.items()}, round_dir))
            (round_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            rounds.append(summary)

            if candidates:
                confirming = k > ratings.ROUNDS
                ledger = _with_seeds(store.load(env, fps=fps), stage_seeds(k))
                for c in list(open_):
                    test = ratings.paired_test(paired(ledger, fps, c, reference, sorted(opponents_so_far))[0])
                    stage = "confirm" if confirming else "screen"
                    print(f"after round {k} ({stage}): {_short_name(c):<28} {_test_line(test)}")
                    v = ratings.decide(test, k - ratings.ROUNDS if confirming else k, confirming)
                    if v is None:
                        continue
                    report.setdefault(c, {})[stage] = _test_dict(test)
                    if v != "confirm":
                        open_.remove(c)
                        report[c].update(verdict=v, rounds=k, decided_in=stage)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    health = rounds[0]
    print_signals(health, f"{run_id}/round_1")
    ledger = store.load(env, fps=fps)
    if candidates:
        print(f"\nVerdicts against {_short_name(reference)} ({played} matches played, {reused} from the store):")
    for c in candidates:
        r = report[c]
        decided = r[r["decided_in"]]
        print(
            f"  {_short_name(c):<28} {r['verdict'].upper():<10} in the {r['decided_in']} after {r['rounds']} "
            f"round(s): {_test_line(ratings.PairedTest(**decided))}"
        )
        upto = _with_seeds(ledger, {seed_of(j) for j in range(1, r["rounds"] + 1)})
        r["per_opponent"] = _per_opponent(upto, fps, c, reference, sorted(opponents_so_far))
        if today in fps and today not in (c, reference):  # not the same code as the reference
            test = ratings.paired_test(paired(upto, fps, c, today, sorted(opponents_so_far - {today}))[0])
            r["vs_todays_champion"] = {"name": today, **_test_dict(test)}
            print(f"    against today's {_short_name(today)} (same shared code): {_test_line(test)}")
        r["stalls"] = stall_coverage(upto, fps, c)
        print(f"    {'opponent':<26} {'candidate GD':>12} {'reference GD':>12} {'matches':>8} (same seed and colour)")
        for o, row in r["per_opponent"].items():
            name = o if o == "head-to-head" else _short_name(o)
            print(f"    {name:<26} {row['candidate']:>+12} {row['champion']:>+12} {row['matches']:>8}")
        for line in stall_lines(r["stalls"], "candidate"):
            print(f"    {line}")

    if failed:
        print(f"\n{len(failed)} match(es) crashed (summary.json 'failed'); their pairs are missing from the verdicts.")
    out = run_dir / "report.json"
    out.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run": _run_metadata(),
                "reference": reference,
                "core": core,
                "environment": env,
                "rule": {"rounds": ratings.ROUNDS, "alpha": ratings.ALPHA, "confirm_seed": CONFIRM_SEED},
                "candidates": report,
                "failed": failed,
                "rounds": [{"round": s["round"], "seed": s["seed"], "matches": len(s["results"])} for s in rounds],
                "signals": {k: v for k, v in health.items() if k not in ("results",)},
            },
            indent=2,
        )
    )
    print(f"\nReport: {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")


# --- the verdict's arithmetic -----------------------------------------------------------------


def _with_seeds(ledger: dict, seeds: set[int]) -> dict:
    """The stored matches with one of `seeds`: a decision uses its own stage's rounds only, whatever
    other seeds the store already holds, so it doesn't depend on earlier runs."""
    return {f: e for f, e in ledger.items() if e["seed"] in seeds}


def _test_dict(test: ratings.PairedTest) -> dict:
    return {"n": test.n, "mean": round(test.mean, 3), "se": round(test.se, 3), "p": round(test.p, 4)}


def _test_line(test: ratings.PairedTest) -> str:
    return f"mean diff {test.mean:+.2f} ± {test.se:.2f} goals/match over {test.n} diffs, p {test.p:.3f}"


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


def paired(ledger: dict, fps: dict, candidate: str, reference: str, opponents: list[str]) -> tuple[list, list]:
    """The candidate's goal difference minus the reference's against each opponent, over the matches
    both played with the same seed and colour (the verdict's diffs), and the candidate's margins
    against the reference (head-to-head, reported only)."""
    diffs = []
    for o in opponents:
        c, ref = _margins(ledger, fps, candidate, o), _margins(ledger, fps, reference, o)
        diffs += [c[k] - ref[k] for k in sorted(set(c) & set(ref))]
    return diffs, list(_margins(ledger, fps, candidate, reference).values())


def _per_opponent(ledger: dict, fps: dict, candidate: str, reference: str, opponents: list[str]) -> dict:
    """Goal difference against each opponent, the candidate's and the reference's, over the matches
    both played with the same seed and colour (so the two columns face the same noise)."""
    rows = {}
    for o in opponents:
        c, ref = _margins(ledger, fps, candidate, o), _margins(ledger, fps, reference, o)
        common = sorted(set(c) & set(ref))
        rows[o] = {
            "candidate": sum(c[k] for k in common),
            "champion": sum(ref[k] for k in common),
            "matches": len(common),
        }
    h2h = _margins(ledger, fps, candidate, reference)
    rows["head-to-head"] = {"candidate": sum(h2h.values()), "champion": -sum(h2h.values()), "matches": len(h2h)}
    return rows


def stall_coverage(ledger: dict, fps: dict, side: str) -> dict:
    """Stalls `side` was checked for over the stored matches at current code: the watchdog only
    watches config_a, so a match counts as checked when `side` was config_a and its record
    has a stall record (None, an older line, is unknown). `total` counts every match of `side`."""
    checked, total, stalls = 0, 0, []
    for e in ledger.values():
        if side not in (e["a"], e["b"]):
            continue
        if e["a_fp"] != fps.get(e["a"]) or e["b_fp"] != fps.get(e["b"]):
            continue
        total += 1
        if e["a"] != side or e.get("stalls") is None:
            continue
        checked += 1
        stalls += [{"a": e["a"], "b": e["b"], "seed": e["seed"], **s} for s in e["stalls"]]
    return {"checked": checked, "total": total, "stalls": stalls}


def stall_lines(cov: dict, name: str) -> list[str]:
    """The printed stall report for one side's `stall_coverage`."""
    head = f"stalls checked in {cov['checked']} of {cov['total']} {name} matches (watchdog covers config_a only): "
    if not cov["checked"]:
        return [head + "unknown"]
    if not cov["stalls"]:
        return [head + "none"]
    lines = [head + str(len(cov["stalls"]))]
    for s in cov["stalls"]:
        diagnosis = f" -- {s['diagnosis']}" if s.get("diagnosis") else ""
        lines.append(
            f"    {_short_name(s['a'])} vs {_short_name(s['b'])} seed {s['seed']}: "
            f"{s['kind']} at t={s['sim_time']:.1f}s{diagnosis}"
        )
    return lines


# --- signals ----------------------------------------------------------------------------------


def signals(summary: dict, by_tag: dict[tuple[str, str], dict], round_dir: Optional[Path]) -> dict:
    """A round's signals (docs/signals.md) from its `summary["results"]` and their stored records
    (by (config_a, config_b) under this run's names): restarts, ball losses, the per-strategy table,
    fouls, stalls and how many matches checked them."""
    rows = summary["results"]
    out: dict = {}
    restarts = [e for r in rows for e in (by_tag.get((r["config_a"], r["config_b"]), {}).get("restarts") or [])]
    out["restarts"] = restart_outcomes.summarise(restarts)
    losses = {
        tag(r["config_a"], r["config_b"]): {**rec["losses"], "match": tag(r["config_a"], r["config_b"])}
        for r in rows
        if (rec := by_tag.get((r["config_a"], r["config_b"]))) and rec.get("losses") is not None
    }
    real_losses = chances = None
    if losses:
        ordered = [losses[t] for t in sorted(losses)]
        out["ball_losses"] = turnover_breakdown.breakdown(ordered)
        if round_dir is not None:
            report = turnover_breakdown.report(summary["run_id"], ordered, summary)
            (round_dir / "ball_losses.md").write_text(report)
        real_losses = {t: turnover_breakdown.real_loss_kinds(r) for t, r in losses.items()}
        chances = {t: r["chances"] for t, r in losses.items() if "chances" in r}
    out["strategies"] = strategy_table(rows, real_losses, chances or {})
    out["fouls"] = foul_table(rows)
    out["stall_incidents"] = stall_incidents(rows)
    out["stalls_checked"] = {
        n: {
            "checked": sum(1 for r in rows if r["config_a"] == n and r["stats"] is not None),
            "matches": sum(1 for r in rows if n in (r["config_a"], r["config_b"])),
        }
        for n in summary["config_names"]
    }
    return out


def stall_incidents(results: list[dict]) -> list[dict]:
    """Stall events over `summary.json`-shaped `results`, with deterministic duplicates merged: the
    same kind, onset tick and duration tick in matches that share a strategy is one freeze seen
    against two opponents that hadn't diverged yet (rsim is deterministic), not two separate
    problems. In order of first appearance."""
    incidents: list[dict] = []
    by_key: dict[tuple, dict] = {}
    for r in results:
        t = tag(r["config_a"], r["config_b"])
        for e in (r.get("stats") or {}).get("stall_events") or []:
            ticks = (e["kind"], round(e["sim_time"] * TICKS_PER_SECOND), round(e["duration_s"] * TICKS_PER_SECOND))
            keys = [(name, *ticks) for name in (r["config_a"], r["config_b"])]
            incident = next((by_key[k] for k in keys if k in by_key), None)
            if incident is None:
                incident = {"kind": e["kind"], "sim_time": e["sim_time"], "duration_s": e["duration_s"], "matches": []}
                incidents.append(incident)
            if t not in incident["matches"]:
                incident["matches"].append(t)
            for k in keys:
                by_key.setdefault(k, incident)
    return incidents


def foul_table(results: list[dict]) -> dict[str, dict]:
    """Every logged foul (`MatchStats.fouls`, both sides) by rule, then by "strategy/tactic" of the
    offending robot. `inferred` counts fouls whose rule only named a team, attributed to that side's
    robot nearest the ball."""
    table: dict[str, dict] = {}
    for r in results:
        for foul in (r.get("stats") or {}).get("fouls") or []:
            strategy = _short_name(r["config_a"] if foul["side"] == "friendly" else r["config_b"])
            row = table.setdefault(foul["rule"], {"total": 0, "inferred": 0, "by_tactic": {}})
            row["total"] += 1
            row["inferred"] += bool(foul["inferred"])
            key = f"{strategy}/{foul['tactic'] or 'unknown'}"
            row["by_tactic"][key] = row["by_tactic"].get(key, 0) + 1
    for row in table.values():
        row["by_tactic"] = dict(sorted(row["by_tactic"].items(), key=lambda kv: -kv[1]))
    return dict(sorted(table.items(), key=lambda kv: -kv[1]["total"]))


def print_signals(summary: dict, run_label: str) -> None:
    """The printed health check: stalls with their coverage, then every signal table."""
    rows = summary["results"]
    stalled = [r for r in rows if (r["stats"] or {}).get("stall_events")]
    backstop = [r for r in rows if r["possession_backstop"] and r not in stalled]
    checked = summary["stalls_checked"]
    print(f"\nSTALLS in round 1: {len(stalled)} match(es), {len(summary['stall_incidents'])} distinct incident(s)")
    print(
        "  checked as config_a (the watchdog watches config_a only): "
        + ", ".join(f"{_short_name(n)} {c['checked']}/{c['matches']}" for n, c in checked.items())
    )
    for r in stalled:
        for e in r["stats"]["stall_events"]:
            tactics = f" tactics={e['tactic_ids']}" if e.get("tactic_ids") else ""
            diagnosis = f" -- {e['diagnosis']}" if e.get("diagnosis") else ""
            print(
                f"  {tag(r['config_a'], r['config_b']):<50} {e['kind']:<17} onset t={e['sim_time']:.1f}s "
                f"referee={e.get('referee_command')}{tactics}{diagnosis}"
            )
    for r in backstop:
        poss = r["stats"]["possession_pct"]
        print(
            f"  {tag(r['config_a'], r['config_b']):<50} {'POSSESSION_BACKSTOP':<17} "
            f"possession {poss['friendly']:.0%}/{poss['enemy']:.0%} ball_travel {r['stats']['ball_travel_m']:.2f}m"
        )
    _print_restarts(summary["restarts"])
    if "ball_losses" in summary:
        _print_ball_losses(summary["ball_losses"], run_label)
    _print_strategy_table(summary["strategies"])
    _print_loss_kinds(summary["strategies"])
    _print_chances(summary["strategies"])
    _print_foul_table(summary["fouls"])
    _print_motion_planning_table(summary["strategies"])


def _print_restarts(restarts: dict) -> None:
    if not restarts["restarts"]:
        return
    print(
        f"\nRESTARTS: {restarts['restarts']}, {restarts['reached_normal_start'] / restarts['restarts']:.0%} "
        "reached NORMAL_START"
    )
    for kind, row in restarts["by_kind"].items():
        outcomes = ", ".join(f"{k} {n}" for k, n in row["outcomes"].items())
        print(f"  {kind:<16} {row['n']:>4}  reached {row['reached_normal_start'] / row['n']:>4.0%}  {outcomes}")


def _print_foul_table(table: dict[str, dict]) -> None:
    if not table:
        return
    print("\nFOULS (both sides; top strategy/tactic of the offending robot; * = rule named only a team):")
    for rule, row in table.items():
        top = ", ".join(f"{k} {c}" for k, c in list(row["by_tactic"].items())[:4])
        star = "*" if row["inferred"] else ""
        print(f"  {rule + star:<30} {row['total']:>5}  {top}")


def _print_strategy_table(table: dict[str, dict]) -> None:
    print("\nSTRATEGIES (per match; losses = real ball losses, as config_a only):")
    print(
        f"  {'strategy':<32} {'W-D-L':>9} {'GF':>5} {'GA':>5} {'shots':>6} {'passes':>7} {'entries':>8} {'fouls':>6} {'losses':>7} {'stalled':>8}"
    )
    for name, t in sorted(table.items()):
        n = max(1, t["matches"])
        losses = f"{t['real_losses_as_a'] / t['matches_as_a']:.1f}" if t["matches_as_a"] else "-"
        print(
            f"  {_short_name(name):<32} {t['wins']:>3}-{t['draws']}-{t['losses']:<3} {t['goals_for'] / n:>5.2f} "
            f"{t['goals_against'] / n:>5.2f} {t['shots'] / n:>6.2f} {t['completed_passes'] / n:>7.1f} "
            f"{t['attacking_third_entries'] / n:>8.1f} {t['fouls'] / n:>6.1f} {losses:>7} {t['stalled']:>8}"
        )


def _print_loss_kinds(table: dict[str, dict]) -> None:
    """Real ball losses per match as config_a, by kind: where each strategy gives the ball away."""
    rows = {name: t for name, t in table.items() if t["matches_as_a"]}
    if not rows:
        return
    totals: dict[str, int] = {}
    for t in rows.values():
        for kind, n in t["real_loss_kinds_as_a"].items():
            totals[kind] = totals.get(kind, 0) + n
    kinds = sorted(totals, key=lambda k: -totals[k])
    short = {
        "shot_saved_or_blocked": "shot_saved",
        "ball_out_after_kick": "out_kick",
        "ball_out_other": "out_other",
        "pass_intercepted": "intercepted",
        "loose_ball_lost": "loose_lost",
    }
    print("\nLOSS KINDS (real ball losses per match as config_a, by kind; see ball_losses.md):")
    print(f"  {'strategy':<32} " + " ".join(f"{short.get(k, k)[:11]:>11}" for k in kinds))
    for name, t in sorted(rows.items(), key=lambda kv: -kv[1]["real_losses_as_a"] / kv[1]["matches_as_a"]):
        n = t["matches_as_a"]
        print(
            f"  {_short_name(name):<32} " + " ".join(f"{t['real_loss_kinds_as_a'].get(k, 0) / n:>11.2f}" for k in kinds)
        )


def _print_chances(table: dict[str, dict]) -> None:
    """Chances created and conceded per strategy, both sides (`analysis.chances`)."""
    rows = {name: t for name, t in table.items() if t["chances"]["matches"]}
    if not rows:
        return

    def pct(v: Optional[float]) -> str:
        return f"{v:.0%}" if v is not None else "-"

    def num(v: Optional[float]) -> str:
        return f"{v:.1f}" if v is not None else "-"

    print(
        "\nCHANCES (both sides; conv = goals per shot, open = goal mouth unblocked at the shot, "
        "regain>shot = open-play regains shot from within 10 s, danger = s/match the enemy held "
        "the ball in our defensive third, fk>shot = free kicks shot from within 10 s, "
        "progress = m gained per completed pass):"
    )
    print(
        f"  {'strategy':<28} {'shots':>6} {'conv':>5} {'dist':>5} {'open':>5} {'save':>5} {'open vs':>7} "
        f"{'regain>shot':>11} {'secs':>5} {'danger':>6} {'fk>shot':>7} {'progress':>8} {'fwd':>4}"
    )
    for name, t in sorted(rows.items()):
        c = t["chances"]
        print(
            f"  {_short_name(name):<28} {c['shots'] / c['matches']:>6.1f} {pct(c['conversion']):>5} "
            f"{num(c['shot_distance_m']):>5} {pct(c['shot_open_goal']):>5} {pct(c['save_rate']):>5} "
            f"{pct(c['faced_open_goal']):>7} {pct(c['regain_to_shot']):>11} {num(c['regain_to_shot_s']):>5} "
            f"{num(c['danger_s_per_match']):>6} {pct(c['free_kick_to_shot']):>7} "
            f"{num(t['pass_progress_m']):>8} {pct(t['forward_pass_share']):>4}"
        )


def _print_motion_planning_table(table: dict[str, dict]) -> None:
    if not any(t.get("worst_clearance_m") is not None or t.get("robot_overlap_ticks", 0) > 0 for t in table.values()):
        return
    print(
        "\nMOTION PLANNING (per match averages; tm_coll = teammate overlap s (incidents), "
        "total_coll = all overlap s (incidents), near_miss = near miss s <5cm, min_clr = worst clearance):"
    )
    print(f"  {'strategy':<32} {'tm_coll':>18} {'total_coll':>20} {'near_miss':>11} {'min_clr':>10}")
    for name, t in sorted(table.items()):
        n = max(1, t["matches"])
        tm = f"{t.get('teammate_overlap_ticks', 0) / n / TICKS_PER_SECOND:.2f}s ({t.get('teammate_overlap_incidents', 0) / n:.1f})"
        tot = f"{t.get('robot_overlap_ticks', 0) / n / TICKS_PER_SECOND:.2f}s ({t.get('robot_overlap_incidents', 0) / n:.1f})"
        near = t.get("near_miss_ticks", 0) / n / TICKS_PER_SECOND
        wc = t.get("worst_clearance_m")
        clr = f"{wc * 100:.1f} cm" if wc is not None else "-"
        print(f"  {_short_name(name):<32} {tm:>18} {tot:>20} {near:>10.2f}s {clr:>10}")


def _print_ball_losses(b: dict, run_label: str) -> None:
    def top(counts: dict[str, int], k: int = 5) -> str:
        return ", ".join(f"{name} {c}" for name, c in list(counts.items())[:k]) or "none"

    print(
        f"\nBALL LOSSES (config_a side): {b['real_losses']} real ({b['real_losses_per_match']}/match; "
        f"{b['raw_turnovers']} raw MatchStats turnovers)"
    )
    print(f"  by kind:   {top(b['by_kind'])}")
    print(f"  fouls:     {top(b['fouls_by_rule'])}")
    print(f"  by tactic: {top(b['by_tactic'])}")
    rec = b.get("receptions")
    if rec and rec["passes"]:
        print(
            f"  passes:    {top(rec['by_outcome'])}; {rec['catch_rate'] or 0:.0%} of reachable caught, "
            f"missed median facing off {rec['missed']['median_facing_off_deg']} deg"
        )
    print(f"  full breakdown: replays/{run_label}/ball_losses.md")


# --- run metadata -----------------------------------------------------------------------------


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _run_metadata() -> dict:
    """What produced this run: enough to reproduce it or line it up against another."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "argv": sys.argv[1:],
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    main()
