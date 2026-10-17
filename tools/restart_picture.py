"""Picture of where the referee override sends every robot for one restart.

    pixi run python tools/restart_picture.py REPLAY.npz --t 260 [--side enemy] [--out PATH]
    pixi run python tools/restart_picture.py REPLAY.npz --all-restarts [--side enemy] [--out PATH]
    pixi run python tools/restart_picture.py REPLAY.npz --findings - [--side enemy]

Rebuilds the field state at `--t` from the replay (`utama_core.replay.scenario`), runs
`RefereeOverride.tick` for the replay's referee command there, exactly as the strategy does, and draws
the target each friendly robot is commanded to: the full pitch, the ball and its 0.8 m keep-out circle,
the placement or free-kick spot, each robot with its id, role (kicker, placer, keeper) and an arrow to its
commanded target, and a text box with the command, which team it is for, and a red flag on any target
inside the keep-out circle or a defence area it must avoid. `--all-restarts` draws one small panel per
distinct restart command in the replay's referee sidecar (the first occurrence of each, at most 8).
`--side enemy` shows the other team's override. Read-only; the planner and the referee are not changed.

`--findings PATH` writes one JSON line per flag (`utama_core/analysis/findings.py`) for every restart in the
match, each occurrence, without drawing: kinds `target_in_keep_out`, `keep_out_short` (the 1 cm stop deadband)
and `target_in_defense_area`. Give `--t` or `--all-restarts` too to set `image` on the findings whose
restart was drawn. `--findings -` writes the JSON to stdout and the text to stderr.

Use it when a restart goes wrong: robots inside the keep-out, a kicker that never reaches the ball, a
placer or defender parked in the wrong place. Filter on the findings first, open the picture to understand
them. See docs/STRATEGY_DEVELOPMENT.md, "Referee handling".
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # keep pygame's banner off stdout (--findings -)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis.findings import Finding, write_jsonl  # noqa: E402
from utama_core.analysis.restart_picture import (  # noqa: E402
    plan_restart,
    render_restart,
    render_restart_grid,
    restart_commands,
    restart_findings,
    restart_game,
)
from utama_core.engine.referee_override import is_override_command  # noqa: E402
from utama_core.replay.scenario import scenario_from_replay  # noqa: E402

MAX_PANELS = 8


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Finding]:
    """Draw what the flags ask for and return the findings of the whole match (when `--findings` is given)."""
    stem = args.replay.stem
    if args.all_restarts:
        commands = restart_commands(args.replay)
        if not commands:
            parser.error("no restart command in the replay's referee sidecar")
        if len(commands) > MAX_PANELS:
            print(f"{len(commands)} restart commands; drawing the first {MAX_PANELS}")
            commands = commands[:MAX_PANELS]
        plans = []
        drawn_t: list[float] = []
        for t, command in commands:
            try:
                scenario = scenario_from_replay(args.replay, t, showing=command)
            except ValueError as err:
                print(f"skipped {command.name} at t={t:g} s: {err}")
                continue
            game = restart_game(args.replay, scenario, args.side)
            plans.append(plan_restart(game, command, sim_time=scenario.sim_time))
            drawn_t.append(t)
        if not plans:
            parser.error("no restart could be rebuilt from the replay")
        out = args.out or REPO_ROOT / "replays" / "restart_picture" / f"{stem}_restarts_{args.side}.png"
        render_restart_grid(plans, out, title=f"{stem}: referee override for each restart ({args.side} side)")
        for plan in plans:
            print(f"{plan.command.name:<26} {plan.side:<7} t={plan.sim_time:g} s  flags: {len(plan.flagged)}")
        print(f"wrote {out}")
        return _with_images(args, {t: out for t in drawn_t})

    scenario = scenario_from_replay(args.replay, args.t)
    command = scenario.referee_command
    if command is None or not is_override_command(command):
        name = command.name if command is not None else "none"
        parser.error(f"referee command at t={args.t:g} s is {name}, not a restart the override handles")
    game = restart_game(args.replay, scenario, args.side)
    plan = plan_restart(game, command, sim_time=scenario.sim_time)
    out = args.out or REPO_ROOT / "replays" / "restart_picture" / f"{stem}_t{args.t:g}_{args.side}.png"
    render_restart(plan, out, title=f"{stem}: {args.side} side's override")
    print(f"wrote {out}")
    for rp in plan.robots:
        target = "-" if rp.target is None else f"({rp.target[0]:+.2f}, {rp.target[1]:+.2f})"
        flags = "; ".join(rp.flags) or "ok"
        print(f"  robot {rp.robot_id:>2} {rp.role or '':<8} {rp.kind:<4} {target:<18} {flags}")
    return _with_images(args, {args.t: out})


def _with_images(args: argparse.Namespace, drawn_at: dict) -> list[Finding]:
    """The whole match's findings (only when `--findings` is given), with `image` set where a picture was drawn."""
    if args.findings is None:
        return []
    findings = restart_findings(args.replay, args.side)
    return [
        dataclasses.replace(f, image=str(drawn_at[f.t0])) if drawn_at.get(f.t0) is not None else f for f in findings
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("replay", type=Path, help="a replay .npz from a run")
    parser.add_argument("--t", type=float, help="time in seconds from the replay's first frame")
    parser.add_argument(
        "--all-restarts",
        action="store_true",
        help="one small panel per distinct restart command in the replay's referee sidecar (at most 8)",
    )
    parser.add_argument(
        "--side",
        choices=["friendly", "enemy"],
        default="friendly",
        help="whose override to draw: friendly is the replay's config_a (default), enemy its config_b",
    )
    parser.add_argument("--out", type=Path, help="PNG path (default: replays/restart_picture/...)")
    parser.add_argument(
        "--findings",
        metavar="PATH",
        help="also write the flagged targets of every restart in the match as JSON lines (- for stdout)",
    )
    args = parser.parse_args()
    if args.findings is None and args.all_restarts == (args.t is not None):
        parser.error("give exactly one of --t or --all-restarts")
    if args.findings is not None and args.t is None and not args.all_restarts:
        # findings only: no picture, the match's findings are all there is to write
        write_jsonl(restart_findings(args.replay, args.side), args.findings)
        return

    if args.findings == "-":
        # stdout carries only the JSON; the picture's text goes to stderr
        with contextlib.redirect_stdout(sys.stderr):
            findings = _run(args, parser)
        write_jsonl(findings, "-")
    else:
        findings = _run(args, parser)
        if args.findings is not None:
            write_jsonl(findings, args.findings)


if __name__ == "__main__":
    main()
