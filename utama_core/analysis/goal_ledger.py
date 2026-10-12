"""Where goals come from: every goal attributed to the situation that started the play it ended.

    pixi run python -m utama_core.analysis.goal_ledger replays/tournament_<id> --out ledger.json

A strategy's goals per match is exactly a sum over situation types of how often the situation
happens per match times how often it ends in a goal (P). That splits a result into what a
change can move: the situations a strategy creates or allows, and how well it converts or
defends them. Only a replay's `.npz` and `.sparse_referee.pkl` are read, so it runs on
converted official logs as well as our own matches.

A **situation** starts when
- a restart goes live: kickoff, free kick or penalty, owned by the kicking side, with the third
  of its spot in that side's attack frame; or a force start, owned by no one;
- a side wins the ball in open play: a possession spell starts, owned by that side, with the
  third it won the ball in, in its own attack frame.

**Possession** in play: the holder is the side with a robot on the ball (`*_has_ball`) while
the other has none; otherwise the previous holder keeps it, and nobody holds it after a
stoppage until someone touches it. A spell shorter than `REGAIN_MIN_HOLD_S` is merged into
the one before it (two robots on one ball flipping). The kicking side's first spell starting
within `FOLD_S` of its restart going live is the restart being taken, not a new situation.

**Attribution**: each goal goes to the latest situation start at or before it and less than
`WINDOW_S` before it, a restart winning a tie with a spell at the same tick; with none, to
"sustained possession". A goal's time is when play stopped for it: our referee scores on that
tick, the official game controller seconds later.

**Two referees.** The official game controller's free-kick command lets the kicker play at
once and stays on until the next stoppage; our `CustomReferee` holds it while the kicker
lines up, then sends NORMAL_START. A match whose referee ever follows a free kick with
NORMAL_START is read the second way, any other the first.

Possession here is not `MatchStats`' (nearest robot within `_POSSESSION_RADIUS_M` of a slow
ball, kept through a pass in flight), and a spell is not one of `chances.py`'s regains (which
skip the first `JUST_RESTARTED_S` after a restart): the ledger needs the moment a side first
touches the ball, for every change of possession, so that its situations partition every goal.
Thirds are `chances.py`'s (`THIRD_DEPTH_M` from either goal line).

Strategy names come from `--labels`, else the run's `summary.json`/`ladder.json` (config_a is
friendly), else the referee's team names (an official log's teams).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from utama_core.analysis.chances import LIVE, REGAIN_MIN_HOLD_S, THIRD_DEPTH_M
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.referee.referee_command import RefereeCommand as C
from utama_core.replay.columnar_reader import load_columnar_replay

WINDOW_S = 10.0
FOLD_S = 1.5
SUSTAINED = "sustained possession"

_LIVE = [c.value for c in LIVE]
# restart command -> (kind, kicking side is yellow)
_RESTARTS = {
    C.PREPARE_KICKOFF_YELLOW.value: ("kickoff", True),
    C.PREPARE_KICKOFF_BLUE.value: ("kickoff", False),
    C.PREPARE_PENALTY_YELLOW.value: ("penalty", True),
    C.PREPARE_PENALTY_BLUE.value: ("penalty", False),
    C.DIRECT_FREE_YELLOW.value: ("free kick", True),
    C.DIRECT_FREE_BLUE.value: ("free kick", False),
    C.INDIRECT_FREE_YELLOW.value: ("free kick", True),
    C.INDIRECT_FREE_BLUE.value: ("free kick", False),
}
_FREE_KICKS = [cmd for cmd, (kind, _) in _RESTARTS.items() if kind == "free kick"]
_PAUSES = [C.STOP.value, C.HALT.value, -1]  # -1: no referee data at that tick
_LADDER_SUFFIX = re.compile(r"_[RL][Kk]$")

_SITUATIONS = [
    f"{kind}, {third} third" for kind in ("ball won", "free kick") for third in ("attacking", "middle", "defensive")
]
TYPES = [f"{owner} {s}" for owner in ("own", "opponent") for s in (*_SITUATIONS, "kickoff", "penalty")]
TYPES += ["force start", SUSTAINED]


def third(x: float) -> str:
    """The pitch third of `x` in the frame of a side attacking +x."""
    half = STANDARD_FIELD_DIMS.full_field_half_length
    if half - x < THIRD_DEPTH_M:
        return "attacking"
    if half + x < THIRD_DEPTH_M:
        return "defensive"
    return "middle"


def live_mask(cmd: np.ndarray) -> np.ndarray:
    """Ticks in play (see "Two referees" above)."""
    free = np.isin(cmd, _FREE_KICKS)
    two_step = bool(np.any(free[:-1] & (cmd[1:] == C.NORMAL_START.value)))
    return np.isin(cmd, _LIVE) | (free & (not two_step))


def find_restarts(ts, cmd, live, ball_x, friendly_attack, my_team_is_yellow: bool) -> list[dict]:
    """Every restart going live: `t`, `kind`, kicking `side` (None for a force start) and the
    `third` of the spot in its frame. `friendly_attack` is +1/-1 per tick, the x friendly attacks."""
    paused = np.isin(cmd, _PAUSES)
    last_command = np.maximum.accumulate(np.where(paused, -1, np.arange(len(cmd))))
    out = []
    for t in np.flatnonzero(live[1:] & ~live[:-1]) + 1:
        # A free kick that is itself live (official controller) restarts play on its own tick;
        # otherwise the restart is the last command before play resumed, unless that was play.
        j = t if int(cmd[t]) in _RESTARTS else last_command[t - 1]
        resumed_play = j < 0 or (j < t and live[j])
        kind, yellow = ("force start", None) if resumed_play else _RESTARTS.get(int(cmd[j]), ("force start", None))
        restart = {"t": float(ts[t]), "kind": kind, "side": None, "third": None}
        if yellow is not None:
            friendly = yellow == my_team_is_yellow
            restart["side"] = "friendly" if friendly else "enemy"
            restart["third"] = third(friendly_attack[t] * (1 if friendly else -1) * ball_x[t])
        out.append(restart)
    return out


def possession_spells(ts, live, friendly_has, enemy_has, ball_x, friendly_attack) -> list[dict]:
    """Open-play possession spells: `side`, `t` (won the ball), `t_end`, and the `third` it won
    the ball in, in its own attack frame. See "Possession" above."""
    ts, live, friendly_has, enemy_has = ts.tolist(), live.tolist(), friendly_has.tolist(), enemy_has.tolist()
    raw = []  # [side (+1 friendly, -1 enemy), first tick, last tick]
    holder = 0
    for t in range(len(ts)):
        if not live[t]:
            holder = 0
            continue
        if friendly_has[t] != enemy_has[t]:
            holder = 1 if friendly_has[t] else -1
        if holder == 0:
            continue
        if raw and raw[-1][0] == holder and raw[-1][2] == t - 1:
            raw[-1][2] = t
        else:
            raw.append([holder, t, t])
    spells = []
    for side, first, last in raw:
        short = ts[last] - ts[first] < REGAIN_MIN_HOLD_S
        joined = bool(spells) and spells[-1][2] + 1 == first
        if joined and (short or spells[-1][0] == side):
            spells[-1][2] = last
        elif not short:  # a flicker with no spell to join, at the start of play, is dropped
            spells.append([side, first, last])
    return [
        {
            "side": "friendly" if side == 1 else "enemy",
            "t": ts[first],
            "t_end": ts[last],
            "third": third(friendly_attack[first] * side * ball_x[first]),
            "takes_restart": False,
        }
        for side, first, last in spells
    ]


def fold_restart_takes(restarts: list[dict], spells: list[dict]) -> None:
    """Mark the kicking side's first spell within `FOLD_S` of its restart (`takes_restart`)."""
    starts = [s["t"] for s in spells]
    for r in restarts:
        i = bisect.bisect_left(starts, r["t"])
        if (
            r["side"] is not None
            and i < len(spells)
            and spells[i]["side"] == r["side"]
            and spells[i]["t"] - r["t"] <= FOLD_S
        ):
            spells[i]["takes_restart"] = True


def situations(match: dict) -> list[dict]:
    """A match's situation starts in time order, a restart after a spell at the same time."""
    starts = [(r["t"], 1, r) for r in match["restarts"]]
    starts += [(s["t"], 0, {**s, "kind": "ball won"}) for s in match["spells"] if not s["takes_restart"]]
    return [s for _, _, s in sorted(starts, key=lambda e: e[:2])]


def attribute(goals: list[dict], starts: list[dict]) -> None:
    """Give each goal its `situation` (None: sustained possession) and `after_s`."""
    times = [s["t"] for s in starts]
    for goal in goals:
        i = bisect.bisect_right(times, goal["t"]) - 1
        hit = i >= 0 and goal["t"] - times[i] < WINDOW_S
        goal["situation"] = {k: starts[i][k] for k in ("t", "kind", "side", "third")} if hit else None
        goal["after_s"] = goal["t"] - times[i] if hit else None


def find_goals(ts, live, sparse_referee: dict, my_team_is_yellow: bool) -> list[dict]:
    """Goals from the referee's score changes (a lowered score withdraws that side's latest goal),
    each timed when play stopped for it."""
    last_live = np.maximum.accumulate(np.where(live, np.arange(len(live)), -1))
    goals: list[dict] = []
    before = None
    for tick in sorted(sparse_referee):
        referee = sparse_referee[tick]
        if referee.yellow_team is None or referee.blue_team is None:
            continue
        score = {True: referee.yellow_team.score, False: referee.blue_team.score}  # yellow?
        if before is not None:
            stopped = tick if live[tick] or last_live[tick] < 0 else last_live[tick] + 1
            for yellow in (True, False):
                side = "friendly" if yellow == my_team_is_yellow else "enemy"
                change = score[yellow] - before[yellow]
                goals += [{"t": float(ts[stopped]), "side": side} for _ in range(change)]
                for _ in range(-change):
                    goals.remove(next(g for g in reversed(goals) if g["side"] == side))
        before = score
    return sorted(goals, key=lambda g: g["t"])


def match_ledger(npz_path, labels: Optional[tuple[str, str]] = None) -> dict:
    """One replay's goals, restarts and possession spells, each goal with its situation.
    `labels` is (friendly, enemy); by default the referee's team names."""
    path = Path(npz_path)
    replay = load_columnar_replay(path)
    ts, cmd = replay.ts, replay.referee_command.astype(np.int64)
    side_is_right = (
        replay.side_is_right if replay.side_is_right is not None else np.full(len(ts), replay.my_team_is_right)
    )
    friendly_attack = np.where(side_is_right, -1.0, 1.0)
    ball_x = replay.ball_p[:, 0]
    live = live_mask(cmd)

    restarts = find_restarts(ts, cmd, live, ball_x, friendly_attack, replay.my_team_is_yellow)
    spells = possession_spells(
        ts, live, replay.friendly_has_ball.any(axis=1), replay.enemy_has_ball.any(axis=1), ball_x, friendly_attack
    )
    fold_restart_takes(restarts, spells)
    match = {"match": path.name[: -len(".npz")], "path": str(path), "restarts": restarts, "spells": spells}
    match["goals"] = find_goals(ts, live, replay.sparse_referee, replay.my_team_is_yellow)
    attribute(match["goals"], situations(match))
    match["labels"] = dict(zip(("friendly", "enemy"), labels or _referee_names(replay)))
    return match


def _referee_names(replay) -> tuple[str, str]:
    referee = next((r for _, r in sorted(replay.sparse_referee.items()) if r.yellow_team is not None), None)
    yellow = (referee.yellow_team.name if referee else "") or "yellow"
    blue = (referee.blue_team.name if referee else "") or "blue"
    return (yellow, blue) if replay.my_team_is_yellow else (blue, yellow)


def run_labels(run_dir: Path) -> dict[str, tuple[str, str]]:
    """Replay stem (without a ladder's `_RK`-style suffix) -> (config_a, config_b), from a
    round-robin's `summary.json` or a ladder's `ladder.json`; empty if there is neither."""
    for name in ("summary.json", "ladder.json"):
        if (run_dir / name).exists():
            results = json.loads((run_dir / name).read_text())["results"]
            return {
                f"{_short(r['config_a'])}_vs_{_short(r['config_b'])}": (r["config_a"], r["config_b"]) for r in results
            }
    return {}


def _short(config: str) -> str:
    """A config's name in replay file names (`match._short_name`)."""
    return config.removeprefix("build_").removesuffix("_kernel_strategy")


def situation_type(situation: Optional[dict], side: str) -> str:
    """`situation`'s name in `TYPES` from `side`'s view."""
    if situation is None:
        return SUSTAINED
    if situation["side"] is None:
        return situation["kind"]
    owner = "own" if situation["side"] == side else "opponent"
    if situation["kind"] in ("kickoff", "penalty"):
        return f"{owner} {situation['kind']}"
    return f"{owner} {situation['kind']}, {situation['third']} third"


def tables(matches: list[dict]) -> dict[str, dict]:
    """Per strategy: matches, goals for and against, and per situation type how often it
    happened per match, P(scored)/P(conceded) from it, and goals scored/conceded per match from
    it. Each goals-per-match column sums to the strategy's goals per match."""
    totals: dict[str, dict] = {}
    for m in matches:
        starts = situations(m)
        for side in ("friendly", "enemy"):
            t = totals.setdefault(
                m["labels"][side],
                {
                    "matches": 0,
                    "situations": collections.Counter(),
                    "scored": collections.Counter(),
                    "conceded": collections.Counter(),
                },
            )
            t["matches"] += 1
            t["situations"].update(situation_type(s, side) for s in starts)
            for g in m["goals"]:
                t["scored" if g["side"] == side else "conceded"][situation_type(g["situation"], side)] += 1
    out = {}
    for label, t in sorted(totals.items()):
        n = t["matches"]
        rows = {}
        for typ in TYPES:
            count = None if typ == SUSTAINED else t["situations"][typ]
            row = {"situations_per_match": None if count is None else count / n}
            for what in ("scored", "conceded"):
                row[f"p_{what}"] = t[what][typ] / count if count else None
                row[f"{what}_per_match"] = t[what][typ] / n
            rows[typ] = row
        out[label] = {
            "matches": n,
            "goals_for": sum(t["scored"].values()),
            "goals_against": sum(t["conceded"].values()),
            "types": rows,
        }
    return out


def ledger(paths: list[Path], labels: Optional[dict[str, tuple[str, str]]] = None, workers: int = 8) -> list[dict]:
    """`match_ledger` over run directories and/or replay files; `labels` (stem -> (friendly,
    enemy)) overrides the run's own names."""
    replays = sorted(q for p in paths for q in (Path(p).glob("*.npz") if Path(p).is_dir() else [Path(p)]))
    names: dict[Path, dict] = {}
    jobs = []
    for npz in replays:
        stem = npz.name[: -len(".npz")]
        if npz.parent not in names:
            names[npz.parent] = run_labels(npz.parent)
        jobs.append((labels or {}).get(stem) or names[npz.parent].get(_LADDER_SUFFIX.sub("", stem)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(match_ledger, replays, jobs))


def _fmt(v: Optional[float], digits: int) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def report(strategies: dict[str, dict]) -> str:
    lines = []
    for label, s in strategies.items():
        n = s["matches"]
        lines += [
            f"== {label}: {n} matches, {s['goals_for'] / n:.2f} scored and {s['goals_against'] / n:.2f} conceded per match",
            f"{'situation':36s} {'per match':>9s} {'P(scored)':>9s} {'scored/m':>8s} {'P(conc.)':>8s} {'conc./m':>7s}",
        ]
        for typ, r in s["types"].items():
            if not (r["situations_per_match"] or r["scored_per_match"] or r["conceded_per_match"]):
                continue
            lines.append(
                f"{typ:36s} {_fmt(r['situations_per_match'], 2):>9s} {_fmt(r['p_scored'], 3):>9s} "
                f"{r['scored_per_match']:8.3f} {_fmt(r['p_conceded'], 3):>8s} {r['conceded_per_match']:7.3f}"
            )
        lines.append("")
    error = max(
        (
            abs(sum(r[f"{what}_per_match"] for r in s["types"].values()) - s[total] / s["matches"])
            for s in strategies.values()
            for what, total in (("scored", "goals_for"), ("conceded", "goals_against"))
        ),
        default=0.0,
    )
    lines.append(f"goals per match = sum over situations (largest error {error:.1e})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="run directories and/or replay .npz files")
    parser.add_argument("--labels", type=Path, help='JSON {"<stem>": ["<friendly>", "<enemy>"]}: who played')
    parser.add_argument("--out", type=Path, help="JSON output (default: goal_ledger.json beside the first input)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    labels = {k: tuple(v) for k, v in json.loads(args.labels.read_text()).items()} if args.labels else None
    matches = ledger(args.paths, labels, args.workers)
    strategies = tables(matches)
    print(report(strategies))
    first = args.paths[0]
    out = args.out or (first if first.is_dir() else first.parent) / "goal_ledger.json"
    out.write_text(json.dumps({"strategies": strategies, "matches": matches}, indent=1))
    print(f"wrote {out}: {len(matches)} matches, {sum(len(m['goals']) for m in matches)} goals")


if __name__ == "__main__":
    main()
