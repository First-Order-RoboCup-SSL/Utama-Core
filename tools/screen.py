#!/usr/bin/env python3
# ruff: noqa: E402
"""screen.py — did a change to one strategy do anything in play? A SCREEN, not a verdict.

    pixi run python tools/screen.py STRATEGY --base REF [--run RUN_DIR] [--starts 60] [--seconds 20]

It plays STRATEGY as it is now (the change: the working tree, shared code included) and as it was at
git ref REF (the base, packed as a `.strategy` file in `replays/eval/frozen/`, as `evaluate.py --frozen`
does) from the same short starts: moments of live play drawn at random from the replays of a run in
which STRATEGY played (by default the latest `replays/tournament_*` that has it), so situations come as
often as they do in matches. Both play our side against the same opponent, the base, under the same
seeded reference noise, so a pair of starts differs only by the change.

- **Fired:** the two starts of a pair diverged (any robot or the ball apart by more than `SAME_M`). A
  change that never fires says nothing; the firing rate is the first thing it reports.
- **Differences** (change minus base), over all starts and over the starts where it fired: goals for,
  goals against, and how far upfield the ball was on average.
- **Call**, by a rule fixed before any result (`call()`; docs/strategy_tools_plan.md, "The screen"):
  `SKIP: did not fire` if fewer than `MIN_FIRED_SHARE` of starts fired; `SKIP: fewer goals in the
  starts` if, over the fired starts, the goal difference is lower by more than 2 standard errors;
  otherwise `WORTH EVALUATING`. Only `evaluate.py` gives a verdict.

It never writes the evaluation ledger, keeps no replays, and plays one strategy only: comparing two
different strategies from short starts is the retired scenario bench, whose per-strategy bias made its
ranking fail. Each run appends a line to `replays/eval/screen_log.jsonl` (`--no-record` skips it): the
screen is on trial, and its calls are checked against `evaluate.py`'s verdicts on the same changes
before it may decide anything.

Starts begin from rest with fresh tactic memory (`tools/repro_from_replay.py`'s limits), so behaviour
that needs a long build-up fires less often here than in matches.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import TICKS_PER_SECOND
from tools.evaluation.sim_settings import reference_noise
from tools.sim_setup import load_side, sim_runner
from utama_core.bundle.strategy_file import BundledStrategy, frozen
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.fingerprint import CodeGraph
from utama_core.replay.scenario import apply_scenario, scenario_from_replay
from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise
from utama_core.strategy import catalog

FROZEN_DIR = Path("replays/eval/frozen")  # shared with evaluate.py --frozen: one file per strategy and commit
LOG_PATH = Path("replays/eval/screen_log.jsonl")

# The design, fixed before any change was screened (docs/strategy_tools_plan.md, "The screen").
STARTS = 60
SECONDS = 20.0
MIN_FIRED_SHARE = 0.10
GOAL_SE = 2.0
SAME_M = 1e-6

LIVE = (RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START)
EDGE_S = 5.0  # no starts this close to a replay's ends
MAX_DRAWS = 50  # tries per start to land on live play


@dataclass(frozen=True)
class Start:
    replay: str
    t: float
    seed: int


@dataclass(frozen=True)
class PairResult:
    start: Start
    fired: bool
    diverged_at_s: Optional[float]
    change: dict  # goals_for, goals_against, upfield_m
    base: dict


# --------------------------------------------------------------------------- pure parts (unit tested)


def diverged_at(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], tol: float = SAME_M) -> Optional[int]:
    """The first tick at which two trajectories (one row of coordinates per tick) are apart by more than
    `tol` in any coordinate, or differ in length; None if they never are."""
    for i, (row_a, row_b) in enumerate(zip(a, b)):
        if len(row_a) != len(row_b) or any(abs(x - y) > tol for x, y in zip(row_a, row_b)):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Mean and its standard error; (0, 0) for none, SE 0 for one."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def goal_difference_changes(results: Sequence[PairResult]) -> list[float]:
    """Per pair: the change's goal difference minus the base's."""
    return [
        (r.change["goals_for"] - r.change["goals_against"]) - (r.base["goals_for"] - r.base["goals_against"])
        for r in results
    ]


def call(n_starts: int, fired_gd_changes: Sequence[float]) -> str:
    """The screen's call: `fired_gd_changes` holds the goal-difference change of each start that fired.

    Screening out a change that would have won loses it for good; letting a useless one through costs
    one evaluation. So it skips only a change that did nothing, or one whose fired starts scored clearly
    worse (by more than `GOAL_SE` standard errors)."""
    if n_starts == 0 or len(fired_gd_changes) < MIN_FIRED_SHARE * n_starts:
        return "SKIP: did not fire"
    mean, se = mean_se(fired_gd_changes)
    if mean + GOAL_SE * se < 0:
        return "SKIP: fewer goals in the starts"
    return "WORTH EVALUATING"


def draw_times(rng: random.Random, duration_s: float, n: int) -> list[float]:
    """`n` times uniformly within a replay of `duration_s`, `EDGE_S` from each end."""
    lo, hi = EDGE_S, max(EDGE_S, duration_s - EDGE_S - SECONDS)
    return [rng.uniform(lo, hi) for _ in range(n)]


# --------------------------------------------------------------------------- choosing starts


def replays_with(run_dir: Path, strategy: str) -> list[Path]:
    """The run's replays in which `strategy` played, those where it was config_a (our side) first. In the
    others a start puts it in its opponent's positions: still a situation of its matches."""
    summary = json.loads((run_dir / "summary.json").read_text())
    name = catalog.plain_name(strategy)
    as_a, as_b = [], []
    for r in summary.get("results", []):
        a, b = catalog.plain_name(r["config_a"]), catalog.plain_name(r["config_b"])
        path = run_dir / f"{a}_vs_{b}.npz"
        if not path.exists():
            continue
        if a == name:
            as_a.append(path)
        elif b == name:
            as_b.append(path)
    return as_a + as_b


MIN_REPLAYS = 3  # a run with fewer replays of the strategy (a `--pair` run) gives too few situations


def latest_run_with(strategy: str, replays: Path = Path("replays")) -> Path:
    """The latest run with at least `MIN_REPLAYS` replays of `strategy`, else the latest with any."""
    runs = [d for d in sorted(replays.glob("tournament_*"), reverse=True) if (d / "summary.json").exists()]
    counts = {d: len(replays_with(d, strategy)) for d in runs}
    for wanted in (MIN_REPLAYS, 1):
        for run_dir in runs:
            if counts[run_dir] >= wanted:
                return run_dir
    raise SystemExit(f"no replays/tournament_* has a replay of {strategy}; give one with --run")


def replay_seconds(path: Path) -> float:
    """How long a replay runs, on the clock `scenario_from_replay` takes."""
    from utama_core.replay.columnar_reader import load_columnar_replay

    ts = load_columnar_replay(path).ts
    return float(ts[-1] - ts[0]) if len(ts) else 0.0


def choose_starts(paths: list[Path], n: int, seed: int) -> list[Start]:
    """`n` moments of live play, the replays taken in turn, each with its own noise seed."""
    rng = random.Random(seed)
    lengths = {path: replay_seconds(path) for path in paths}
    starts: list[Start] = []
    for i in range(n):
        path = paths[i % len(paths)]
        for t in draw_times(rng, lengths[path], MAX_DRAWS):
            try:
                scenario = scenario_from_replay(path, t)
            except (ValueError, IndexError):
                continue
            if scenario.referee_command in LIVE:
                starts.append(Start(path.as_posix(), round(t, 3), zlib.crc32(f"{seed}:{i}".encode())))
                break
    return starts


# --------------------------------------------------------------------------- playing


def _play(side: str, opponent: str, start: Start, seconds: float) -> tuple[list[tuple], dict]:
    """One start: our side `side` against `opponent`. Returns the trajectory and the measures."""
    strategy, enemy = load_side(side), load_side(opponent)
    runner, _ = sim_runner(strategy, enemy, sim_noise=SimNoise.from_dict(reference_noise(start.seed)))
    trajectory: list[tuple] = []
    ball_x: list[float] = []
    try:
        apply_scenario(runner, scenario_from_replay(Path(start.replay), start.t), verify=False)
        for _ in range(int(seconds * TICKS_PER_SECOND)):
            runner.step_once()
            frame = runner.my.current_game_frame
            row = []
            if frame.ball is not None:
                row += [frame.ball.p.x, frame.ball.p.y]
                ball_x.append(frame.ball.p.x)
            for robots in (frame.friendly_robots, frame.enemy_robots):
                for rid in sorted(robots):
                    row += [robots[rid].p.x, robots[rid].p.y]
            trajectory.append(tuple(row))
        referee = runner.my.game.referee
        goals_for, goals_against = referee.yellow_team.score, referee.blue_team.score  # our side is yellow
    finally:
        runner.close()
        for s in (strategy, enemy):
            if isinstance(s, BundledStrategy):
                s.close()
    # Our side defends the right goal (sim_runner), so upfield is towards -x.
    upfield = -statistics.fmean(ball_x) if ball_x else 0.0
    return trajectory, {"goals_for": goals_for, "goals_against": goals_against, "upfield_m": round(upfield, 4)}


def play_pair(change: str, base: str, start: Start, seconds: float) -> PairResult:
    change_traj, change_m = _play(change, base, start, seconds)
    base_traj, base_m = _play(base, base, start, seconds)
    tick = diverged_at(change_traj, base_traj)
    return PairResult(start, tick is not None, None if tick is None else tick / TICKS_PER_SECOND, change_m, base_m)


# --------------------------------------------------------------------------- report


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def report(results: list[PairResult], n_starts: int) -> tuple[list[str], dict]:
    fired = [r for r in results if r.fired]
    lines = [f"fired in {len(fired)} of {n_starts} starts"]
    numbers: dict = {"starts": n_starts, "fired": len(fired)}
    for label, group in (("all starts", results), ("fired starts", fired)):
        rows = {
            "goal difference": goal_difference_changes(group),
            "goals for": [r.change["goals_for"] - r.base["goals_for"] for r in group],
            "goals against": [r.change["goals_against"] - r.base["goals_against"] for r in group],
            "ball upfield (m)": [r.change["upfield_m"] - r.base["upfield_m"] for r in group],
        }
        lines.append(f"change minus base, {label} ({len(group)}):")
        for name, values in rows.items():
            mean, se = mean_se(values)
            lines.append(f"  {name:<18} {mean:+.3f} ± {se:.3f} (SE)")
            numbers.setdefault(label, {})[name] = [round(mean, 4), round(se, 4)]
    if fired:
        times = sorted(r.diverged_at_s for r in fired)
        lines.append(f"first divergence: median {statistics.median(times):.2f} s into a start")
    numbers["call"] = call(n_starts, goal_difference_changes(fired))
    return lines, numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("strategy", help="the one strategy the change is to (a catalog name)")
    parser.add_argument("--base", required=True, help="git ref of the code before the change, e.g. HEAD~1 or v2.5.0")
    parser.add_argument("--run", type=Path, default=None, help="run dir to draw starts from (default: latest with it)")
    parser.add_argument("--starts", type=int, default=STARTS)
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--seed", type=int, default=0, help="which starts and noise (default 0)")
    parser.add_argument("--max-workers", type=int, default=8, help="parallel starts (default 8)")
    parser.add_argument("--no-record", action="store_true", help=f"don't append to {LOG_PATH}")
    args = parser.parse_args()

    name = catalog.plain_name(args.strategy)
    if name not in catalog.names():
        raise SystemExit(f"unknown strategy {args.strategy!r}; strategies: {', '.join(catalog.names())}")
    designed = (args.starts, args.seconds) == (STARTS, SECONDS)
    print("SCREEN, not a verdict: did this change do anything in play? Only evaluate.py judges a strategy.")
    if not designed:
        print(
            f"  (off-design: {args.starts} starts of {args.seconds:g} s; the call's rule assumes {STARTS} of {SECONDS:g})"
        )

    base_file = frozen(name, args.base, FROZEN_DIR).as_posix()
    graph = CodeGraph()
    change_fp, base_fp = graph.strategy_fingerprint(name)[:16], graph.strategy_fingerprint(base_file)[:16]
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": name,
        "base": base_file,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "change_fp": change_fp,
        "base_fp": base_fp,
        "seed": args.seed,
        "on_design": designed,
    }
    print(f"  {name} now ({change_fp}) against {base_file} ({base_fp})")

    if change_fp == base_fp:
        lines, numbers = ["the strategy's code is the same at both refs: nothing to play"], {
            "starts": 0,
            "fired": 0,
            "call": "SKIP: did not fire",
        }
    else:
        run_dir = args.run or latest_run_with(name)
        paths = replays_with(run_dir, name)
        starts = choose_starts(paths, args.starts, args.seed)
        record["run"] = run_dir.as_posix()
        print(f"  {len(starts)} starts of {args.seconds:g} s of live play from {len(paths)} replays in {run_dir}")
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            results = list(pool.map(play_pair, [name] * len(starts), [base_file] * len(starts), starts,
                                    [args.seconds] * len(starts)))  # fmt: skip
        lines, numbers = report(results, len(starts))
        record["pairs"] = [asdict(r) for r in results]

    print("\n".join(lines))
    print(f"call: {numbers['call']}")
    record.update(numbers)
    if not args.no_record:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"recorded in {LOG_PATH}")


if __name__ == "__main__":
    main()
