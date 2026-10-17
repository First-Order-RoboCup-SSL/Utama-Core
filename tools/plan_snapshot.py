"""Picture of what the fast-path planner decided for one robot on one tick.

    pixi run python tools/plan_snapshot.py REPLAY.npz --t 260 --robot 4 --target -1.0 -0.3 --side enemy

Rebuilds the field state at `--t` from the replay (`utama_core.replay.scenario`), runs the planner's
`_path_to` for the robot toward `--target`, and writes a PNG: the pitch around the robot, every robot
at its real radius with its id, the ball and its keep-out circle while the planner keeps out of it,
each robot's clearance ring, the target, the carrot the planner returned, and a text box with the
referee command, distances and whether the robot is inside the keep-out circle. Read-only; the
planner is not changed. `--side enemy` plans the other strategy's robot (the replay's enemy). With
no `--target`, the robot's own position 0.5 s later stands in for it, and the text box says so.

Use it when a robot does something odd near an obstacle or a keep-out zone (see
docs/STRATEGY_DEVELOPMENT.md, "Debugging a foul or failure"). The picture is one tick: pair it with
`tools/replay_trace.py` for the timeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis.plan_snapshot import (  # noqa: E402
    game_from_scenario,
    plan_snapshot,
    scenario_robot_position,
    title_for,
)
from utama_core.replay.scenario import scenario_from_replay  # noqa: E402

STAND_IN_DELAY_S = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("replay", type=Path, help="a replay .npz (or .pkl) from a run")
    parser.add_argument("--t", type=float, required=True, help="time in seconds from the replay's first frame")
    parser.add_argument("--robot", type=int, required=True, help="robot id on the planned side")
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="target in the replay's pitch coordinates (metres); default: the robot's position 0.5 s later",
    )
    parser.add_argument(
        "--side",
        choices=["friendly", "enemy"],
        default="friendly",
        help="which strategy's robot to plan: friendly is the replay's config_a (default), enemy its config_b",
    )
    parser.add_argument(
        "--out", type=Path, help="PNG path (default: replays/plan_snapshot/<replay>_t<t>_<side><id>.png)"
    )
    parser.add_argument("--half-width", type=float, default=1.5, help="window half-width around the robot, metres")
    parser.add_argument("--full-pitch", action="store_true", help="draw the whole pitch instead of the window")
    args = parser.parse_args()

    scenario = scenario_from_replay(args.replay, args.t)
    game = game_from_scenario(scenario, args.side)
    planned = scenario.friendly_robots if args.side == "friendly" else scenario.enemy_robots
    if not any(s.id == args.robot for s in planned):
        ids = sorted(s.id for s in planned)
        parser.error(f"no {args.side} robot {args.robot} at t={args.t} s (ids: {ids})")

    note = None
    if args.target is not None:
        target = tuple(args.target)
    else:
        target = scenario_robot_position(args.replay, args.t + STAND_IN_DELAY_S, args.side, args.robot)
        note = f"target: stand-in, robot at t+{STAND_IN_DELAY_S:g} s"

    out = args.out or (
        REPO_ROOT / "replays" / "plan_snapshot" / f"{args.replay.stem}_t{args.t:g}_{args.side}{args.robot}.png"
    )
    snap = plan_snapshot(
        game,
        args.robot,
        target,
        out,
        half_width=args.half_width,
        full_pitch=args.full_pitch,
        title=title_for(scenario, args.side, args.robot),
        target_note=note,
    )

    print(f"wrote {snap.path}")
    print(f"carrot: ({snap.carrot[0]:.3f}, {snap.carrot[1]:.3f})  robot -> carrot {snap.robot_to_carrot:.3f} m")
    if snap.robot_to_ball is not None:
        print(f"robot -> ball {snap.robot_to_ball:.3f} m  inside keep-out: {snap.inside_keep_out}")
    print(f"keep-out active: {snap.keeps_off_ball}  referee: {scenario.referee_command}")


if __name__ == "__main__":
    main()
