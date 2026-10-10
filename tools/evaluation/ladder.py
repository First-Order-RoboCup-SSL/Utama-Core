"""ladder.py — one candidate strategy against a reference pool, instead of a full round-robin.

Run:
    pixi run python tools/evaluation/ladder.py CANDIDATE [flags]

    --pool A B ...        opponents (short or factory names); default: the top POOL_SIZE of the
                          newest complete round-robin's standings, the candidate and
                          round_robin.RETIRED left out
    --standings PATH      the round-robin to take the pool from (its summary.json or run dir)
    --no-reuse            play every match, even ones stored in replays/match_cache/
    --max-workers N, -h/--help

Every pairing is played in mirrored pairs: the same `run_match` settings (which side
config_a starts on, who kicks off) once with the candidate as config_a and once with the
opponent, so each team gets the other's start and side luck cancels. The four settings, in
order, give at most 8 matches per pairing; the first is the round-robin's own (config_a right,
config_a kicks off), so a match a round-robin stored is reused, not played. rsim is
deterministic: playing a setting twice gives the same match, so 8 is all a pairing has until
matches vary by sampled world (docs/roadmap.md item 2a).

Stopping rule (`decided`): after each mirrored pair, a pairing stops once at least
MIN_MATCHES are played and the candidate's wins minus losses is at least LEAD either way, or
can no longer change sign over the matches left. Draws count for neither.

Reports win/draw/loss, points (3 a win, 1 a draw) and goal difference per opponent and in
total, and writes replays/ladder_<id>/ladder.json beside the matches' replays. Played
matches are stored in the match cache in the round-robin's format, so a round-robin reuses
them too. Unlike `round_robin.py --reuse`, reused matches are not spot-checked.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import _short_name, run_match
from tools.evaluation.round_robin import (
    MAX_MATCH_SECONDS,
    RETIRED,
    _resolve_config_names,
    _run_metadata,
)
from utama_core.analysis import restart_outcomes, turnover_breakdown
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.replay import match_cache
from utama_core.replay.fingerprint import CodeGraph, match_key

# (a_is_right, a_kicks_off), in the order a pairing plays them; the first is the round-robin's.
SETTINGS = ((True, True), (True, False), (False, True), (False, False))
MIN_MATCHES = 4
LEAD = 3
POOL_SIZE = 5


def decided(wins: int, losses: int, played: int, max_matches: int) -> bool:
    """Whether a pairing can stop: see the module docstring's stopping rule."""
    margin = abs(wins - losses)
    return played >= max_matches or (played >= MIN_MATCHES and (margin >= LEAD or margin > max_matches - played))


def standings(results: list[dict]) -> dict[str, dict]:
    """Points (3 a win, 1 a draw), goal difference and matches per strategy, over
    `summary.json`-shaped `results`."""
    table: dict[str, dict] = {}
    for r in results:
        for name, gf, ga in ((r["config_a"], r["score_a"], r["score_b"]), (r["config_b"], r["score_b"], r["score_a"])):
            row = table.setdefault(name, {"points": 0, "goal_difference": 0, "matches": 0})
            row["points"] += 3 if gf > ga else 1 if gf == ga else 0
            row["goal_difference"] += gf - ga
            row["matches"] += 1
    return table


def reference_pool(results: list[dict], candidate: str, size: int = POOL_SIZE) -> list[str]:
    """The `size` best strategies in `results` other than `candidate` and the retired ones, by
    points per match, then goal difference per match, then name."""
    table = standings(results)
    ranked = sorted(
        (n for n in table if n != candidate and n not in RETIRED),
        key=lambda n: (
            -table[n]["points"] / table[n]["matches"],
            -table[n]["goal_difference"] / table[n]["matches"],
            n,
        ),
    )
    return ranked[:size]


def latest_round_robin(replays: Path) -> Optional[Path]:
    """The newest `tournament_*/summary.json` under `replays` that played every pair of its
    configs at least once without restart fuzzing: a full round-robin, not a `--pair` run."""
    for path in sorted(replays.glob("tournament_*/summary.json"), reverse=True):
        summary = json.loads(path.read_text())
        n = len(summary.get("config_names", []))
        if n >= 3 and summary.get("fuzz_seed") is None and len(summary.get("results", [])) >= n * (n - 1) // 2:
            return path
    return None


def fixtures(candidate: str, opponent: str, setting: tuple[bool, bool]) -> list[tuple[str, str, bool, bool]]:
    """One mirrored pair: `setting` with the candidate as config_a, then with the opponent."""
    return [(candidate, opponent, *setting), (opponent, candidate, *setting)]


def _suffix(a_is_right: bool, a_kicks_off: bool) -> str:
    """The match file tag's suffix (`_RK`, `_Lk`, ...), except none for the
    round-robin's own setting so its records and file names match a round-robin's."""
    if a_is_right and a_kicks_off:
        return ""
    return f"_{'R' if a_is_right else 'L'}{'K' if a_kicks_off else 'k'}"


def play(a: str, b: str, a_is_right: bool, a_kicks_off: bool, run_dir: Path) -> dict:
    """Play one fixture and analyse its replay into a match-cache record, as `round_robin.py`
    stores them."""
    suffix = _suffix(a_is_right, a_kicks_off)
    result = run_match(
        a,
        b,
        duration_seconds=MAX_MATCH_SECONDS,
        a_is_right=a_is_right,
        a_kicks_off=a_kicks_off,
        run_dir=run_dir,
        match_tag_suffix=suffix,
    )
    replay = run_dir / f"{_short_name(a)}_vs_{_short_name(b)}{suffix}.npz"
    return {
        "result": dataclasses.asdict(result),
        "restarts": restart_outcomes.analyse_match(replay),
        "losses": turnover_breakdown.analyse_match(str(replay)),
    }


def outcome(result: dict, candidate: str) -> tuple[int, int]:
    """The candidate's goals for and against in a stored `MatchResult` dict."""
    if result["config_a"] == candidate:
        return result["score_a"], result["score_b"]
    return result["score_b"], result["score_a"]


def summarise(matches: list[dict], candidate: str) -> dict:
    """W-D-L, points and goal difference over a pairing's (or the ladder's) matches."""
    goals = [outcome(m["result"], candidate) for m in matches]
    wins = sum(gf > ga for gf, ga in goals)
    draws = sum(gf == ga for gf, ga in goals)
    n = len(goals)
    return {
        "matches": n,
        "wins": wins,
        "draws": draws,
        "losses": n - wins - draws,
        "points": 3 * wins + draws,
        "points_per_match": round((3 * wins + draws) / n, 2) if n else None,
        "goal_difference": sum(gf - ga for gf, ga in goals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate")
    parser.add_argument("--pool", nargs="+")
    parser.add_argument("--standings", type=Path)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = parser.parse_args()

    candidate = _resolve_config_names([args.candidate])[0]
    rr_table: dict[str, dict] = {}
    if args.pool:
        pool = [n for n in _resolve_config_names(args.pool) if n != candidate]
    else:
        path = args.standings or latest_round_robin(REPLAY_BASE_PATH)
        if path is not None and path.is_dir():
            path = path / "summary.json"
        if path is None or not path.exists():
            raise SystemExit("No complete round-robin under replays/: pass --pool or --standings")
        results = json.loads(path.read_text())["results"]
        pool = reference_pool(results, candidate)
        rr_table = standings(results)
        print(f"Pool: the top {len(pool)} of {path.parent.name}, by points per match")

    run_id = datetime.now(timezone.utc).strftime("ladder_%Y%m%d_%H%M%S")
    run_dir = REPLAY_BASE_PATH / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = match_cache.MatchCache()
    graph = CodeGraph()

    def key(fixture: tuple[str, str, bool, bool]) -> str:
        a, b, right, kicks = fixture
        return match_key(graph, a, b, duration_seconds=MAX_MATCH_SECONDS, a_is_right=right, a_kicks_off=kicks)

    max_matches = 2 * len(SETTINGS)
    matches: dict[str, list[dict]] = {opp: [] for opp in pool}
    next_setting = {opp: 0 for opp in pool}
    pending = {opp: 0 for opp in pool}
    played: dict[tuple, dict] = {}  # fixture -> record, for the cache
    reused = 0
    keys: dict[tuple, str] = {}
    futures: dict = {}

    print(f"Ladder: {_short_name(candidate)} vs {', '.join(_short_name(o) for o in pool)}; run dir replays/{run_id}/\n")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:

        def advance(opp: str) -> None:
            nonlocal reused
            while pending[opp] == 0:
                s = summarise(matches[opp], candidate)
                if next_setting[opp] == len(SETTINGS) or decided(s["wins"], s["losses"], s["matches"], max_matches):
                    return
                for fixture in fixtures(candidate, opp, SETTINGS[next_setting[opp]]):
                    keys[fixture] = key(fixture)
                    record = None if args.no_reuse else cache.get(keys[fixture])
                    if record is not None:
                        reused += 1
                        matches[opp].append(record)
                    else:
                        futures[executor.submit(play, *fixture, run_dir)] = (opp, fixture)
                        pending[opp] += 1
                next_setting[opp] += 1

        for opp in pool:
            advance(opp)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                opp, fixture = futures.pop(future)
                record = future.result()
                played[fixture] = record
                matches[opp].append(record)
                pending[opp] -= 1
                r = record["result"]
                print(
                    f"  {_short_name(r['config_a']):<24} {r['score_a']} - {r['score_b']} {_short_name(r['config_b']):<24} "
                    f"{'right' if fixture[2] else 'left':<5} {'kicks off' if fixture[3] else ''}",
                    flush=True,
                )
                advance(opp)

    graph = CodeGraph()
    if all(key(f) == keys[f] for f in played):
        for fixture, record in played.items():
            cache.put(keys[fixture], record)
    else:
        print("\nThe code changed while the ladder played; storing nothing in the match cache")

    rows = {opp: summarise(matches[opp], candidate) for opp in pool}
    total = summarise([m for opp in pool for m in matches[opp]], candidate)
    print(f"\n{_short_name(candidate)}: {len(played)} played, {reused} reused from the match cache")
    print(f"  {'opponent':<28} {'rr pts/m':>8} {'W-D-L':>7} {'pts/m':>6} {'GD':>4} {'matches':>8}")
    for opp, s in rows.items():
        rr = rr_table.get(opp)
        rr_ppm = f"{rr['points'] / rr['matches']:.2f}" if rr else "-"
        print(
            f"  {_short_name(opp):<28} {rr_ppm:>8} {s['wins']:>3}-{s['draws']}-{s['losses']:<1} "
            f"{s['points_per_match']:>6.2f} {s['goal_difference']:>+4} {s['matches']:>8}"
        )
    print(
        f"  {'total':<28} {'':>8} {total['wins']:>3}-{total['draws']}-{total['losses']:<1} "
        f"{total['points_per_match']:>6.2f} {total['goal_difference']:>+4} {total['matches']:>8}"
    )

    (run_dir / "ladder.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run": _run_metadata(),
                "candidate": candidate,
                "pool": pool,
                "rule": {"settings": SETTINGS, "min_matches": MIN_MATCHES, "lead": LEAD},
                "opponents": rows,
                "total": total,
                "played": len(played),
                "reused": reused,
                "results": [
                    {k: v for k, v in m["result"].items() if k != "stats"} for opp in pool for m in matches[opp]
                ],
            },
            indent=2,
        )
    )
    print(f"\nFull results: replays/{run_id}/ladder.json")


if __name__ == "__main__":
    main()
