#!/usr/bin/env python3
# ruff: noqa: E402
"""Triage a run dir: which problems happened in which matches, with the numbers, and a picture for the worst.

    pixi run python tools/triage.py REPLAYS_OR_RUN_DIR [--strategy NAME ...] [--matches 3] [--out DIR]
                                    [--max-pictures 8] [--workers 2] [--restarts]

Start here when a strategy plays badly. For each strategy the friendly side (config_a, the only side the
intentions log covers) of its `--matches` worst matches by goal difference is analysed:

- `tactic_thrash` (`match_timeline.thrash_findings`): a robot reverting a tactic within a second, over the whole match;
- `stall` (`stuck_detector.find_stuck_windows`): a window with the ball frozen while a friendly robot oscillates,
  or a restart held far too long (`restart_stall`); for each oscillating robot, `plan_trace` over the window and
  its `parked` / `pushed_out` findings. The planner's target there is the robot's final position in the window
  (the replay does not record the strategy's target), so a robot standing at its target is never `parked`;
- `target_in_keep_out`, `keep_out_short`, `target_in_defense_area` (`restart_picture.restart_findings`): only
  with `--restarts`, which is slow (about 45 s a match).

Every finding is written to `OUT/findings.jsonl`. The `--max-pictures` (default 8) findings that have a picture
source are ranked by kind (stall, target_in_keep_out, target_in_defense_area, parked, pushed_out, tactic_thrash,
keep_out_short), then by value; those get a picture: a tactic_thrash gets `match_timeline` over 40 s from 5 s
before the robot's first flip-back, a stall or parked finding gets its `plan_trace`, a restart finding gets
`restart_picture` at its restart. `OUT/index.md` is the report to read first: per strategy, its matches with
their scores, the finding counts by kind, then each pictured finding with its message and picture path.

`OUT` defaults to `RUN_DIR/triage/`. Read-only: nothing is played or changed.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import multiprocessing
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import match_timeline
from tools.evaluation.match import _short_name
from tools.plan_trace import replay_game_at
from utama_core.analysis.findings import Finding, write_jsonl
from utama_core.analysis.matches import STEM, Match, load_match
from utama_core.analysis.plan_trace import plan_findings, plan_trace
from utama_core.analysis.restart_picture import (
    plan_restart,
    render_restart,
    restart_findings,
    restart_game,
    restart_occurrences,
)
from utama_core.analysis.stuck_detector import find_stuck_windows
from utama_core.engine.referee_override import is_override_command
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.scenario import scenario_from_replay

SIDE = "friendly"
# The match_timeline picture for a tactic_thrash: this much before the robot's first flip-back, this long.
THRASH_LEAD_S = 5.0
THRASH_WINDOW_S = 40.0
# Picture priority, most useful first. pushed_out is a parked sibling (the planner pushed out of a keep-out).
KIND_PRIORITY = (
    "stall",
    "target_in_keep_out",
    "target_in_defense_area",
    "parked",
    "pushed_out",
    "tactic_thrash",
    "keep_out_short",
)


@dataclass(frozen=True)
class ThrashPicture:
    """A match_timeline picture of one window."""

    npz: Path
    t0: float
    t1: float


@dataclass(frozen=True)
class RestartPicture:
    """A restart_picture of the friendly override for one restart command."""

    npz: Path
    t: float
    command: RefereeCommand


@dataclass
class Item:
    """One finding and how to picture it: `png` is already drawn (plan_trace), or `request` is drawn later."""

    finding: Finding
    request: Optional[Union[ThrashPicture, RestartPicture]] = None
    png: Optional[Path] = None
    window: Optional[tuple[float, float]] = (
        None  # the seconds the picture shows (the finding's own t0-t1 may be the match)
    )
    robot: Optional[int] = None  # the robot a plan_trace picture follows
    image: Optional[Path] = None  # set by draw_pictures for the findings that were drawn
    note: Optional[str] = None  # why a picture was not drawn, when one was asked for

    @property
    def pictureable(self) -> bool:
        return self.png is not None or self.request is not None


@dataclass
class MatchReport:
    match: Match
    items: list[Item]
    notes: list[str]


# --- match selection -------------------------------------------------------------------------------


def select_matches(run_dir: Path, strategies: Optional[set[str]], n: int) -> list[Match]:
    """Each strategy's `n` worst matches as config_a (friendly), by its goal difference, worst first.

    `strategies` (short or full config names) limits the selection to those strategies; None means every
    strategy in the dir. A match with no score sorts last."""
    wanted = None if strategies is None else {_short_name(s) for s in strategies}
    by_strategy: dict[str, list[Match]] = defaultdict(list)
    for npz in sorted(run_dir.glob("*.npz")):
        m = STEM.match(npz.name.removesuffix(".npz"))
        if m is None or (wanted is not None and m["a"] not in wanted):
            continue
        match = load_match(npz)
        if match is not None:
            by_strategy[match.a].append(match)
    chosen: list[Match] = []
    for name in sorted(by_strategy):
        ranked = sorted(
            by_strategy[name],
            key=lambda m: (m.goal_difference(name) is None, m.goal_difference(name) or 0, m.npz.name),
        )
        chosen.extend(ranked[:n])
    return chosen


# --- per-match analysis (runs in a worker) ---------------------------------------------------------


def _stall_items(npz: Path, window, game_at, work: Path, notes: list[str]) -> list[Item]:
    """The stall finding for one stuck window, with its plan_trace picture and parked / pushed_out findings."""
    dur = window.t_end - window.t_start
    if window.kind == "restart_stall":
        commands = restart_occurrences(npz)
        held = [c for t, c in commands if t <= window.t_start]
        command = held[-1] if held else None
        name = command.name if command is not None else "a restart command"
        finding = Finding(
            tool="stuck_detector",
            kind="stall",
            replay=str(npz),
            side=SIDE,
            t0=window.t_start,
            t1=window.t_end,
            robots=(),
            value=dur,
            threshold=None,
            unit="s",
            message=f"{name} held for {dur:.1f} s ({window.t_start:.1f}-{window.t_end:.1f} s) with the ball frozen: "
            "no restart ceremony takes that long.",
        )
        ask = RestartPicture(npz, window.t_start, command) if command and is_override_command(command) else None
        return [Item(finding, request=ask, window=(window.t_start, window.t_start))]

    robots = tuple(window.oscillating_robot_ids)
    finding = Finding(
        tool="stuck_detector",
        kind="stall",
        replay=str(npz),
        side=SIDE,
        t0=window.t_start,
        t1=window.t_end,
        robots=robots,
        value=dur,
        threshold=None,
        unit="s",
        message=f"friendly robots {list(robots)} oscillate with the ball frozen (ball std {window.ball_std:.3f} m) "
        f"for {dur:.1f} s ({window.t_start:.1f}-{window.t_end:.1f} s).",
    )
    items: list[Item] = []
    first_png: Optional[Path] = None
    first_rid: Optional[int] = None
    for rid in robots:
        try:
            final = game_at(window.t_end).friendly_robots[rid].p
            png = work / f"{npz.stem}_plan_{window.t_start:g}-{window.t_end:g}_R{rid}.png"
            trace = plan_trace(
                game_at,
                rid,
                window.t_start,
                window.t_end,
                png,
                target=(final.x, final.y),
                title=f"{npz.stem}: R{rid}, {window.t_start:.1f}-{window.t_end:.1f} s (target: final position)",
            )
        except (ValueError, KeyError) as err:
            notes.append(f"plan_trace for R{rid} over {window.t_start:.1f}-{window.t_end:.1f} s failed: {err}")
            continue
        if first_png is None:
            first_png, first_rid = trace.path, rid
        items.extend(
            Item(f, png=trace.path, robot=rid) for f in plan_findings(trace.rows, rid, replay=str(npz), side=SIDE)
        )
    items.insert(0, Item(finding, png=first_png, robot=first_rid))
    for item in items:
        item.window = (window.t_start, window.t_end)
    return items


def analyse_match(match: Match, restarts: bool, work: Path) -> MatchReport:
    """Every finding of one match's friendly side, each with the picture it would get. Draws the plan_trace
    pictures into `work` (the caller keeps the ones that are shown); the others are drawn later."""
    npz = match.npz
    notes: list[str] = []
    items: list[Item] = []

    window = match_timeline.load_window(npz, None, None, SIDE)
    end = float(window.tw[-1])
    for finding in match_timeline.thrash_findings(window, replay=str(npz), side=SIDE):
        rid = finding.robots[0]
        flips = match_timeline.flip_backs(window.lanes[window.robot_ids.index(rid)])
        t0 = max(0.0, flips[0] - THRASH_LEAD_S)
        t1 = min(end, t0 + THRASH_WINDOW_S)
        items.append(Item(finding, request=ThrashPicture(npz, t0, t1) if t1 > t0 else None, window=(t0, t1)))

    # One memoised game_at for every stall window and robot: each call rebuilds the state from the replay (0.3 s),
    # and the robots of one window would otherwise rebuild the same ticks.
    game_at = functools.lru_cache(maxsize=1024)(replay_game_at(npz, SIDE))
    for stuck in find_stuck_windows(npz):
        items.extend(_stall_items(npz, stuck, game_at, work, notes))

    if restarts:
        commands = dict(restart_occurrences(npz))
        for finding in restart_findings(npz, SIDE):
            command = commands.get(finding.t0)
            items.append(
                Item(
                    finding,
                    request=RestartPicture(npz, finding.t0, command) if command else None,
                    window=(finding.t0, finding.t0),
                )
            )
    return MatchReport(match, items, notes)


# --- pictures (main process) -----------------------------------------------------------------------


def _priority(kind: str) -> int:
    return KIND_PRIORITY.index(kind) if kind in KIND_PRIORITY else len(KIND_PRIORITY)


def rank(items: list[Item]) -> list[Item]:
    """The items that have a picture source, most important first: kind priority, then value (high first)."""
    return sorted(
        (it for it in items if it.pictureable),
        key=lambda it: (_priority(it.finding.kind), -it.finding.value),
    )


def _draw_restart(picture: RestartPicture, out: Path) -> Path:
    scenario = scenario_from_replay(picture.npz, picture.t, showing=picture.command)
    game = restart_game(picture.npz, scenario, SIDE)
    plan = plan_restart(game, picture.command, sim_time=scenario.sim_time)
    title = f"{picture.npz.stem}: friendly override for {picture.command.name} at t={picture.t:g} s"
    return render_restart(plan, out, title=title)


def draw_pictures(items: list[Item], pictures: Path, max_pictures: int) -> list[Item]:
    """Give the top `max_pictures` pictureable items an `image` (copied or drawn into `pictures`), and return them.
    Items that share a picture source share one file."""
    drawn: dict[tuple, Path] = {}
    shown = rank(items)[:max_pictures]
    for rank_no, item in enumerate(shown, start=1):
        if item.png is not None:
            key: tuple = ("png", str(item.png))
        elif isinstance(item.request, ThrashPicture):
            r = item.request
            key = ("thrash", str(r.npz), r.t0, r.t1)
        else:
            r = item.request
            key = ("restart", str(r.npz), r.t)
        if key not in drawn:
            stem = Path(key[1]).stem
            dst = pictures / f"{rank_no:02d}_{item.finding.kind}_{stem}.png"
            try:
                if item.png is not None:
                    shutil.copyfile(item.png, dst)
                elif key[0] == "thrash":
                    match_timeline.build(item.request.npz, item.request.t0, item.request.t1, SIDE, dst)
                else:
                    _draw_restart(item.request, dst)
                drawn[key] = dst
            except Exception as exc:  # noqa: BLE001 - a picture is optional; the finding is not
                item.note = f"picture failed: {type(exc).__name__}: {exc}"
                continue
        item.image = drawn[key]
    return shown


# --- report ----------------------------------------------------------------------------------------


def _span_of(window: Optional[tuple[float, float]]) -> str:
    if window is None:
        return "the match"
    a, b = window
    return f"{a:.1f}-{b:.1f} s" if b > a else f"{a:.1f} s"


def _span(f: Finding) -> str:
    return _span_of((f.t0, f.t1)) if f.t0 is not None and f.t1 is not None else "the match"


def write_index(
    out: Path,
    run_dir: Path,
    reports: list[MatchReport],
    max_pictures: int,
    restarts: bool,
) -> Path:
    lines = [
        f"# Triage: {run_dir.name}",
        "",
        f"Friendly side (config_a) of each strategy's worst matches by goal difference. "
        f"Pictures: the top {max_pictures} findings with a picture source. Restart findings: "
        f"{'yes' if restarts else 'no (--restarts not given)'}.",
        "",
    ]
    by_strategy: dict[str, list[MatchReport]] = defaultdict(list)
    for r in reports:
        by_strategy[r.match.a].append(r)
    if not reports:
        lines.append("No matches with this strategy as config_a in the run dir.")
    for name in sorted(by_strategy):
        group = by_strategy[name]
        lines += [f"## {name}", "", "Matches analysed (score friendly-enemy, goal difference):"]
        for r in group:
            m = r.match
            score = "no score" if m.score_a is None else f"{m.score_a}-{m.score_b}"
            gd = m.goal_difference(name)
            lines.append(f"- {m.npz.stem}: {score} (GD {gd if gd is not None else 'n/a'})")
        for note in sorted({n for r in group for n in r.notes}):
            lines.append(f"- note: {note}")
        counts = Counter(it.finding.kind for r in group for it in r.items)
        lines += ["", "| kind | findings |", "|---|---|"]
        if not counts:
            lines.append("| (none) | 0 |")
        for kind, n in sorted(counts.items(), key=lambda kv: _priority(kv[0])):
            lines.append(f"| {kind} | {n} |")
        lines.append("")
        pictured = [it for r in group for it in r.items if it.image is not None or it.note]
        if not pictured:
            lines += ["No pictured findings.", ""]
        for it in pictured:
            f = it.finding
            lines += [f"### {f.kind} at {_span(f)}", "", f.message]
            if it.image is not None:
                who = f" (R{it.robot})" if it.robot is not None else ""
                lines.append(f"Picture of {_span_of(it.window)}{who}: `{it.image.resolve()}`")
            if it.note:
                lines.append(f"Picture not drawn: {it.note}")
            lines.append("")
    path = out / "index.md"
    path.write_text("\n".join(lines))
    return path


def triage(
    run_dir: Union[str, Path],
    *,
    strategies: Optional[set[str]] = None,
    matches: int = 3,
    out: Optional[Union[str, Path]] = None,
    restarts: bool = False,
    max_pictures: int = 8,
    workers: int = 2,
) -> Path:
    """Analyse the selected matches, draw the top pictures and write `out/index.md` and `out/findings.jsonl`.
    Returns the index path."""
    run_dir = Path(run_dir)
    out = Path(out) if out is not None else run_dir / "triage"
    work, pictures = out / "_work", out / "pictures"
    for d in (work, pictures):
        d.mkdir(parents=True, exist_ok=True)
    selected = select_matches(run_dir, strategies, matches)
    jobs = [(m, restarts, work) for m in selected]
    if workers > 1 and len(jobs) > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            reports = pool.starmap(analyse_match, jobs)
    else:
        reports = [analyse_match(*j) for j in jobs]
    items = [it for r in reports for it in r.items]
    draw_pictures(items, pictures, max_pictures)
    shutil.rmtree(work, ignore_errors=True)
    write_jsonl(
        [dataclasses.replace(it.finding, image=str(it.image) if it.image else None) for it in items],
        out / "findings.jsonl",
    )
    return write_index(out, run_dir, reports, max_pictures, restarts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir", type=Path, help="a round-robin dir (replays/tournament_<id>) or any dir of match .npz files"
    )
    parser.add_argument(
        "--strategy",
        action="append",
        metavar="NAME",
        help="only matches where this strategy is config_a (repeatable; short or build_*_kernel_strategy names)",
    )
    parser.add_argument(
        "--matches", type=int, default=3, help="each strategy's N worst matches by goal difference (default 3)"
    )
    parser.add_argument("--out", type=Path, default=None, help="output dir (default RUN_DIR/triage/)")
    parser.add_argument("--max-pictures", type=int, default=8, help="pictures to draw (default 8)")
    parser.add_argument("--workers", type=int, default=2, help="worker processes, one match each (default 2)")
    parser.add_argument(
        "--restarts",
        action="store_true",
        help="also check every restart's referee override (restart_picture findings). SLOW: about 45 s a match",
    )
    args = parser.parse_args()
    index = triage(
        args.run_dir,
        strategies=set(args.strategy) if args.strategy else None,
        matches=args.matches,
        out=args.out,
        restarts=args.restarts,
        max_pictures=args.max_pictures,
        workers=args.workers,
    )
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
