"""The fast-path planner over a time window: one robot's plan, tick by tick.

    pixi run python tools/plan_trace.py REPLAY.npz --t0 250 --t1 262 --robot 4 --side enemy --target -1.0 -0.3

For each sampled time in [t0, t1] (every `--step` s) it rebuilds the field state from the replay
(`utama_core.replay.scenario`), runs the planner's `_path_to` for the robot, and records the robot's
position, the carrot returned, the referee command, the distance to the ball, the clearance and the
speed. It writes one PNG: a pitch panel with the robot's path coloured by time and each sample's
carrot joined to the robot, above time panels for robot-to-ball distance (with the keep-out
distance), robot-to-carrot length, speed and the referee command. Read-only; the planner is not
changed. Without `--target` each tick's target is the robot's position `--lookahead` s later, the
stand-in `tools/plan_snapshot.py` uses; the figure says so. `--json` prints the per-sample rows.

Use it when a robot's behaviour near an obstacle or keep-out zone unfolds over time, where one
`plan_snapshot` tick cannot show it (see docs/STRATEGY_DEVELOPMENT.md, "Debugging a foul or failure").
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis.plan_snapshot import game_from_scenario  # noqa: E402
from utama_core.analysis.plan_trace import plan_trace  # noqa: E402
from utama_core.replay.scenario import scenario_from_replay  # noqa: E402


def replay_game_at(replay: Path, side: str):
    """`game_at(t)` for `plan_trace`: the `Game` the planned side sees in the replay at `t` seconds."""

    def game_at(t: float):
        return game_from_scenario(scenario_from_replay(replay, t), side)

    return game_at


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("replay", type=Path, help="a replay .npz (or .pkl) from a run")
    parser.add_argument("--t0", type=float, required=True, help="window start, seconds from the replay's first frame")
    parser.add_argument("--t1", type=float, required=True, help="window end, seconds from the replay's first frame")
    parser.add_argument("--robot", type=int, required=True, help="robot id on the planned side")
    parser.add_argument(
        "--side",
        choices=["friendly", "enemy"],
        default="friendly",
        help="which strategy's robot to plan: friendly is the replay's config_a (default), enemy its config_b",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="fixed target in the replay's pitch coordinates (metres); default: the robot's position --lookahead s later",
    )
    parser.add_argument("--step", type=float, default=0.1, help="sampling interval, seconds (default 0.1)")
    parser.add_argument(
        "--lookahead", type=float, default=0.5, help="stand-in delay when no --target is given, seconds (default 0.5)"
    )
    parser.add_argument(
        "--out", type=Path, help="PNG path (default: replays/plan_trace/<replay>_<t0>-<t1>_<side><id>.png)"
    )
    parser.add_argument("--json", action="store_true", help="print the per-sample rows as JSON")
    args = parser.parse_args()

    if args.t1 < args.t0:
        parser.error("--t1 is before --t0")
    if args.step <= 0:
        parser.error("--step must be positive")

    first = scenario_from_replay(args.replay, args.t0)
    planned = first.friendly_robots if args.side == "friendly" else first.enemy_robots
    if not any(s.id == args.robot for s in planned):
        parser.error(f"no {args.side} robot {args.robot} at t={args.t0} s (ids: {sorted(s.id for s in planned)})")

    out = args.out or (
        REPO_ROOT / "replays" / "plan_trace" / f"{args.replay.stem}_{args.t0:g}-{args.t1:g}_{args.side}{args.robot}.png"
    )
    config = first.config_a_name if args.side == "friendly" else first.config_b_name
    short = (config or "?").removeprefix("build_").removesuffix("_kernel_strategy")
    title = f"planned {args.side} {short}, robot {args.robot}, t = {args.t0:g} to {args.t1:g} s"

    trace = plan_trace(
        replay_game_at(args.replay, args.side),
        args.robot,
        args.t0,
        args.t1,
        out,
        step=args.step,
        target=tuple(args.target) if args.target is not None else None,
        lookahead=args.lookahead,
        title=title,
    )

    if args.json:
        print(json.dumps([dataclasses.asdict(r) for r in trace.rows], indent=1))
    else:
        print(f"wrote {trace.path}")
        print(f"{len(trace.rows)} samples, {trace.target_note}")
        max_offset = max(trace.rows, key=lambda r: r.carrot_offset)
        min_carrot = min(trace.rows, key=lambda r: r.robot_to_carrot)
        print(
            f"max carrot offset from the robot-to-target line: {max_offset.carrot_offset:.3f} m at t={max_offset.t:g}"
        )
        print(f"min robot-to-carrot: {min_carrot.robot_to_carrot:.3f} m at t={min_carrot.t:g}")


if __name__ == "__main__":
    main()
