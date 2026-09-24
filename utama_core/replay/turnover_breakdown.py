"""Breakdown of *how* the friendly team loses the ball, over a tournament run's replays.

`smoke_tournament.py` runs this after every saved run: headline numbers go into
`summary.json` under `ball_losses`, the full report into `ball_losses.md` beside it. For an
existing run:

    pixi run python -m utama_core.replay.turnover_breakdown replays/tournament_<id> --out report.md

Why: across a round-robin, turnovers outnumber completed passes more than 2:1 and play
rarely reaches the attacking third, but `MatchStats` only counts turnovers. This says which
kind of loss dominates and which tactic had the ball, so a fix lands where it matters.

Two kinds of loss are reported:

1. **Turnovers** — exactly `MatchStats`' own `turnovers` counter (the possession-radius state
   machine in `utama_core/engine/match_stats.py`). Each replay is fed through a real
   `MatchStatsAccumulator`; every tick its friendly turnover counter increments is classified.
   Totals therefore match each match's recorded `turnovers` by construction — the report
   checks this and prints any mismatch.
2. **Restarts given away** — play stopped while we held (or last held) the ball and the next
   restart went to the opponent: ball out of bounds, or a foul. The state machine never
   counts these as turnovers.

Friendly is always `config_a` (yellow): `tournament_lib.run_match` writes the intentions log,
used for tactic attribution, for that side only.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.match_stats import _POSSESSION_RADIUS_M, MatchStatsAccumulator
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.replay_player import load_frames_in_range

_LIVE = {RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START}
# Restarts that hand the ball to blue (the enemy). Kickoffs are excluded: blue kicks off
# after *we* score, which is not a loss.
_ENEMY_RESTARTS = {
    RefereeCommand.BALL_PLACEMENT_BLUE,
    RefereeCommand.DIRECT_FREE_BLUE,
    RefereeCommand.INDIRECT_FREE_BLUE,
    RefereeCommand.PREPARE_PENALTY_BLUE,
}
# A robot still this close to the ball when the opponent takes control was tackled; farther
# away, it had already lost the ball and failed to win it back.
_TACKLE_RADIUS_M = 1.5 * _POSSESSION_RADIUS_M
_JUST_RESTARTED_S = 3.0
# A turnover we win back this fast is almost always the nearest-robot flipping between two
# robots pressed on the same ball, not a real loss of possession.
_FLICKER_S = 1.0

TURNOVER_KINDS = {
    "pass_intercepted": "released at speed, not goalward; opponent controlled it next",
    "shot_saved_or_blocked": "released at speed toward the goal mouth; opponent controlled it next",
    "tackled": "opponent took control while our holder was still within 1.5x possession radius",
    "loose_ball_lost": "our holder had drifted off the ball (no kick); opponent reached it first",
    "during_stoppage": "possession changed while play was stopped (e.g. opponent placing the ball)",
}
RESTART_KINDS = {
    "ball_out_after_kick": "ball left the field after our last kick; restart to opponent",
    "ball_out_other": "ball left the field without a kick from us (dribbled/deflected out)",
    "foul": "play stopped with the ball in the field; restart to opponent",
}


def _rule_name(status_message: str) -> str:
    """'Excessive dribbling: 1.01m > 1.0m' / 'Double touch by yellow 3' -> the rule alone."""
    return re.split(r":| by ", status_message, maxsplit=1)[0].strip()


def _goalward(x: float, y: float, vx: float, vy: float, attack_sign: float) -> bool:
    """Same on-target geometry as `MatchStatsAccumulator`'s shot detector, without its speed gate."""
    if vx * attack_sign <= 0:
        return False
    goal_x = attack_sign * STANDARD_FIELD_DIMS.full_field_half_length
    y_at_goal = y + vy * (goal_x - x) / vx
    return abs(y_at_goal) <= STANDARD_FIELD_DIMS.half_goal_width


class _TacticTimeline:
    """robot id -> tactic class name at a given sim time, forward-filled from intention rows."""

    def __init__(self, intentions_path: Path):
        self._events = []
        if intentions_path.exists():
            with open(intentions_path) as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("event") == "intention":
                        self._events.append(row)
        self._events.sort(key=lambda r: r["sim_time"])
        self._i = 0
        self._slots: dict[str, tuple[str, frozenset[int]]] = {}

    def tactic_at(self, t: float, robot_id: int) -> str:
        while self._i < len(self._events) and self._events[self._i]["sim_time"] <= t:
            row = self._events[self._i]
            if row["tactic_id"] == "referee reset":
                self._slots.clear()
            else:
                self._slots[row["tactic_id"]] = (row.get("note") or row["tactic_id"], frozenset(row["robot_ids"]))
            self._i += 1
        for name, robots in self._slots.values():
            if robot_id in robots:
                return name
        return "unassigned"


def analyse_match(npz_path: str) -> dict:
    """Classify every friendly ball loss in one replay. Returns plain dicts (picklable)."""
    path = Path(npz_path)
    stem = path.name[: -len(".npz")]
    frames = load_frames_in_range(path, 0.0, math.inf)
    tactics = _TacticTimeline(path.with_name(f"{stem}.intentions.jsonl"))
    acc = MatchStatsAccumulator()

    turnovers: list[dict] = []
    restarts: list[dict] = []
    release: Optional[dict] = None  # our last speed release of the ball
    live_since = -math.inf
    prev_cmd = None
    pending_stop: Optional[dict] = None  # a live->stopped transition awaiting its restart
    unresolved: list[dict] = []  # turnovers we haven't won the ball back from yet

    half_len = STANDARD_FIELD_DIMS.full_field_half_length
    half_wid = STANDARD_FIELD_DIMS.full_field_half_width

    for frame in frames:
        cmd = frame.referee.referee_command if frame.referee else None
        attack_sign = -1.0 if frame.my_team_is_right else 1.0
        keeper_id = frame.referee.yellow_team.goalkeeper if frame.referee else 0

        def who(robot_id: int, t: float, command) -> str:
            if command not in _LIVE:
                return "restart override"
            if robot_id == keeper_id:
                return "GoalkeeperTactic"
            return tactics.tactic_at(t, robot_id)

        if cmd in _LIVE and prev_cmd not in _LIVE:
            live_since = frame.ts
        if prev_cmd in _LIVE and cmd is not None and cmd not in _LIVE and frame.ball is not None:
            pending_stop = {
                "t": frame.ts,
                "ball_out": abs(frame.ball.p.x) > half_len or abs(frame.ball.p.y) > half_wid,
                "we_had_it": acc._poss_side == "friendly",
                "last_was_kick": acc._poss_robot_id is None and release is not None,
                "robot": acc._poss_robot_id if acc._poss_robot_id is not None else (release or {}).get("robot"),
                "rule": None,
            }
        if pending_stop is not None and pending_stop["rule"] is None and frame.referee and frame.referee.status_message:
            pending_stop["rule"] = _rule_name(frame.referee.status_message)
        if pending_stop is not None and cmd not in (None, RefereeCommand.STOP, RefereeCommand.HALT):
            if cmd in _ENEMY_RESTARTS and pending_stop["we_had_it"]:
                if pending_stop["ball_out"]:
                    kind = "ball_out_after_kick" if pending_stop["last_was_kick"] else "ball_out_other"
                else:
                    kind = "foul"
                robot = pending_stop["robot"]
                restarts.append(
                    {
                        "kind": kind,
                        "t": pending_stop["t"],
                        "rule": pending_stop["rule"] or "unknown",
                        "tactic": (
                            (release or {}).get("tactic")
                            if pending_stop["last_was_kick"]
                            else (
                                who(robot, pending_stop["t"] - 1e-6, RefereeCommand.NORMAL_START)
                                if robot is not None
                                else "unknown"
                            )
                        ),
                    }
                )
            pending_stop = None

        prev_side, prev_robot = acc._poss_side, acc._poss_robot_id
        prev_turnovers = acc._turnovers
        acc.record_tick(frame)
        if acc._poss_side == "friendly":
            for t in unresolved:
                t["regained_after_s"] = frame.ts - t["t"]
            unresolved.clear()

        if prev_side == "friendly" and prev_robot is not None and acc._poss_robot_id is None and frame.ball is not None:
            v = frame.ball.v
            release = {
                "robot": prev_robot,
                "t": frame.ts,
                "goalward": _goalward(frame.ball.p.x, frame.ball.p.y, v.x, v.y, attack_sign),
                "tactic": who(prev_robot, frame.ts, cmd),
                "just_restarted": frame.ts - live_since <= _JUST_RESTARTED_S,
            }

        if acc._turnovers > prev_turnovers:
            if cmd not in _LIVE:
                kind, tactic, just = "during_stoppage", "restart override", False
            elif prev_robot is None and release is not None:
                kind = "shot_saved_or_blocked" if release["goalward"] else "pass_intercepted"
                tactic, just = release["tactic"], release["just_restarted"]
            else:
                holder = frame.friendly_robots.get(prev_robot) if prev_robot is not None else None
                close = holder is not None and holder.p.distance_to(frame.ball.p) <= _TACKLE_RADIUS_M
                kind = "tackled" if close else "loose_ball_lost"
                tactic = who(prev_robot, frame.ts, cmd) if prev_robot is not None else "unknown"
                just = frame.ts - live_since <= _JUST_RESTARTED_S
            entry = {"kind": kind, "t": frame.ts, "tactic": tactic, "just_restarted": just, "regained_after_s": None}
            turnovers.append(entry)
            unresolved.append(entry)

        prev_cmd = cmd

    return {"match": stem, "turnovers": turnovers, "restarts": restarts, "acc_turnovers": acc._turnovers}


def _table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def analyse_run(run_dir: Path, workers: int = 8) -> list[dict]:
    """`analyse_match` over every replay in a tournament run directory."""
    paths = sorted(str(p) for p in Path(run_dir).glob("*.npz"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(analyse_match, paths))


def _is_real(turnover: dict) -> bool:
    """Not a stoppage handover (already counted as the foul/ball-out restart) and not won back
    within `_FLICKER_S` (two robots on one ball; the nearest robot flips)."""
    regained = turnover["regained_after_s"]
    return turnover["kind"] != "during_stoppage" and not (regained is not None and regained <= _FLICKER_S)


def breakdown(results: list[dict]) -> dict:
    """Headline numbers for `summary.json`: real losses, and how, by which rule, by which tactic."""
    n = len(results)
    real = [t for r in results for t in r["turnovers"] if _is_real(t)]
    rss = [x for r in results for x in r["restarts"]]
    losses = real + rss

    def counts(key: str, items: list[dict]) -> dict[str, int]:
        return dict(collections.Counter(x[key] for x in items).most_common())

    return {
        "matches": n,
        "raw_turnovers": sum(len(r["turnovers"]) for r in results),
        "real_losses": len(losses),
        "real_losses_per_match": round(len(losses) / n, 2) if n else 0.0,
        "by_kind": counts("kind", losses),
        "fouls_by_rule": counts("rule", [x for x in rss if x["kind"] == "foul"]),
        "by_tactic": counts("tactic", losses),
    }


def report(run_name: str, results: list[dict], summary: dict) -> str:
    recorded = {f"{r['config_a']}_vs_{r['config_b']}": r["stats"].get("turnovers") for r in summary["results"]}

    def short(name: str) -> str:
        return name.removeprefix("build_").removesuffix("_kernel_strategy")

    recorded = {"_vs_".join(short(p) for p in k.split("_vs_")): v for k, v in recorded.items()}
    mismatches = [
        (r["match"], r["acc_turnovers"], recorded.get(r["match"]))
        for r in results
        if recorded.get(r["match"]) != r["acc_turnovers"]
    ]

    n = len(results)
    tos = [t for r in results for t in r["turnovers"]]
    rss = [x for r in results for x in r["restarts"]]
    total = len(tos) + len(rss)
    analysed = {r["match"] for r in results}
    passes = sum(
        r["stats"].get("completed_passes", 0)
        for r in summary["results"]
        if f"{short(r['config_a'])}_vs_{short(r['config_b'])}" in analysed
    )

    lines = [
        f"# Turnover breakdown — `{run_name}`",
        "",
        f"{n} matches, friendly = `config_a` (yellow). {len(tos)} turnovers + {len(rss)} restarts given "
        f"away = **{total} ball losses** ({total / n:.1f}/match), against {passes} completed passes "
        f"({passes / n:.1f}/match).",
        "",
        "Turnover counts match each match's recorded `MatchStats.turnovers`: "
        + (
            "**yes, all matches**."
            if not mismatches
            else f"**no — {len(mismatches)} mismatches** (first: {mismatches[:3]})."
        ),
        "",
        "## How the ball is lost",
        "",
    ]
    kinds = collections.Counter(t["kind"] for t in tos) + collections.Counter(x["kind"] for x in rss)
    just = collections.Counter(t["kind"] for t in tos if t["just_restarted"])
    flicker = collections.Counter(
        t["kind"] for t in tos if t["regained_after_s"] is not None and t["regained_after_s"] <= _FLICKER_S
    )
    desc = {**TURNOVER_KINDS, **RESTART_KINDS}
    rows = [
        [
            f"`{k}`",
            c,
            f"{c / total:.0%}",
            f"{c / n:.2f}",
            f"{flicker[k] / c:.0%}" if k in TURNOVER_KINDS else "",
            just.get(k, "") or "",
            desc[k],
        ]
        for k, c in kinds.most_common()
    ]
    lines.append(
        _table(
            rows,
            [
                "kind",
                "count",
                "share",
                "per match",
                f"won back ≤{_FLICKER_S:.0f}s",
                f"≤{_JUST_RESTARTED_S:.0f}s after restart",
                "meaning",
            ],
        )
    )
    real = [t for t in tos if _is_real(t)]
    lines += [
        "",
        f"**Real losses: {len(real) + len(rss)}** ({(len(real) + len(rss)) / n:.1f}/match) — excluding "
        f"turnovers won back within {_FLICKER_S:.0f}s (two robots on one ball; the nearest-robot flips) "
        "and `during_stoppage` (the opponent handling the ball for a restart already counted as "
        "`foul`/`ball_out_*`).",
    ]

    fouls = [x for x in rss if x["kind"] == "foul"]
    lines += ["", "## Which rule the `foul` restarts were for", ""]
    by_rule = collections.defaultdict(collections.Counter)
    for x in fouls:
        by_rule[x["rule"]][x["tactic"]] += 1
    rows = [
        [f"{rule}", sum(c.values()), ", ".join(f"`{t}` {k}" for t, k in c.most_common(3))]
        for rule, c in sorted(by_rule.items(), key=lambda kv: -sum(kv[1].values()))
    ]
    lines.append(_table(rows, ["rule (referee status message)", "count", "top tactics"]))

    lines += ["", "## Which tactic had the ball (real losses only)", ""]
    by_tactic = collections.defaultdict(collections.Counter)
    total = len(real) + len(rss)
    kinds = collections.Counter(t["kind"] for t in real) + collections.Counter(x["kind"] for x in rss)
    for x in real + rss:
        by_tactic[x["tactic"]][x["kind"]] += 1
    cols = [k for k, _ in kinds.most_common()]
    rows = []
    for tactic, c in sorted(by_tactic.items(), key=lambda kv: -sum(kv[1].values())):
        s = sum(c.values())
        rows.append([f"`{tactic}`", s, f"{s / total:.0%}"] + [c.get(k, 0) for k in cols])
    lines.append(_table(rows, ["tactic", "losses", "share"] + [f"`{k}`" for k in cols]))
    lines += [
        "",
        "`restart override` = the robot was driven by `RefereeOverride` (restart positioning), not a tactic.",
        "Ball-out and foul losses are attributed to the tactic of our last robot on the ball.",
        "`unassigned` = no tactic slot held the robot at that moment.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    results = analyse_run(args.run_dir, args.workers)
    md = report(args.run_dir.name, results, json.loads((args.run_dir / "summary.json").read_text()))
    if args.out:
        args.out.write_text(md)
        print(f"wrote {args.out}")
    print(md)


if __name__ == "__main__":
    main()
