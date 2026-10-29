# ruff: noqa: E402
"""examples/demos.py — every dashboard demo from one command, switched from the Live tab.

Run:
    pixi run demos                       # real robots, starting with the shooting drill
    pixi run demos --demo pass_receive   # another first
    pixi run demos --mode rsim           # rehearse in the simulator

One dashboard stays up (http://localhost:8080). The Live tab's Demo panel lists the demos; Halt,
pick one and press Run: the robots stop, the running demo closes (its log saved, radio and vision
released) and the chosen one starts, with the page reloading by itself. Each demo is its example
(`DEMOS`), built as when run on its own: on real robots each robot's lid is checked against vision
first (`shooting_drill.on_field`), so a robot found again turns a little on the spot. Within a
demo the Strategy panel still switches tactics. Ctrl+C stops the robots and ends it.

The panel also holds:
- the settings changed most between runs (`SETTINGS`), applied from the next Run: which robot
  plays first (the drill's attacker, the shooter, robot A), the part of the field the cameras
  see, and the referee's rules;
- on real robots, Field check: `pixi run field-check` (vision, then each robot turned on the
  spot to pair it with its lid) with the radio freed for it, its lines shown there, then the
  same demo again;
- "teleop" (`teleop.py`): drive either robot by keyboard to place robots and the ball.

`planner_ab.py` and `calibrate_robot.py` are not here: they measure, with their own loop and
terminal prompts, rather than wait on the dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import argparse
import signal
import traceback

import field_check
import pass_receive
import save_and_counter
import score_goal
import shooting_drill
import teleop

from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import demo as demo_view

# name -> build its runner on the shared dashboard: (mode, server) -> StrategyRunner
DEMOS = {
    "shooting_drill": lambda mode, server: shooting_drill.build_runner(mode, server=server),
    "pass_receive": lambda mode, server: pass_receive.build_runner(mode, server=server),
    "save_and_counter": lambda mode, server: save_and_counter.build_runner(mode, server=server),
    "score_goal": lambda mode, server: score_goal.build_runner(mode, server=server),
    "teleop": lambda mode, server: teleop.build_runner(mode, server=server),
}

ROBOTS = {"radio 0": shooting_drill.RADIO_0, "radio 1": shooting_drill.RADIO_1}
CAMERAS = ("full", "left", "right")


def _now_first(value, values) -> list:
    return [value] + [v for v in values if v != value]


# setting -> its values, the examples' own config first (the default)
SETTINGS = {
    "first robot": _now_first(next(n for n, r in ROBOTS.items() if r == shooting_drill.ATTACKER), list(ROBOTS)),
    "camera": _now_first(shooting_drill.CAMERA_HALF, CAMERAS),
    "rules": _now_first(shooting_drill.RULES, ("off", "on")),
}


def apply_settings(settings: dict) -> None:
    """Sets the examples' config blocks as an operator would edit them; read by the next build.
    save_and_counter always plays the full field (both goals)."""
    first = ROBOTS[settings["first robot"]]
    second = next(r for r in ROBOTS.values() if r != first)
    shooting_drill.ATTACKER, shooting_drill.DEFENDER = first, second
    score_goal.SHOOTER = first
    for module in (pass_receive, save_and_counter, teleop):
        module.ROBOT_A, module.ROBOT_B = first, second
    for module in (shooting_drill, score_goal, pass_receive, teleop):
        module.CAMERA_HALF = settings["camera"]
    shooting_drill.RULES = settings["rules"]  # every example's `_referee_profile` reads it


def run_check(switcher: demo_view.DemoSwitcher) -> None:
    """`pixi run field-check` on the demos' radios, each line to the terminal and the page."""

    def out(line: str) -> None:
        print(line, flush=True)
        switcher.check_line(line)

    try:
        field_check.check([r.firmware_id for r in ROBOTS.values()], out=out)
    except Exception as e:  # the radio not plugged in, say: shown, and the demo starts again
        out(f"!! field check failed: {e}")


def play(
    mode: str,
    first: str,
    server,
    switcher: demo_view.DemoSwitcher,
    demos=DEMOS,
    configure=apply_settings,
    check=run_check,
) -> None:
    """Plays `first`, then each demo the page asks for (after the field check, when asked), until
    one is stopped otherwise (Ctrl+C)."""
    views = server.views()  # the dashboard's own; each demo's are dropped when it ends
    name, checking = first, False
    while name is not None or checking:
        signal.signal(signal.SIGINT, signal.default_int_handler)  # a runner sets its own while it runs
        if checking:
            print("\n=== field check ===")
            check(switcher)
            switcher.checked()
            if name is None:  # no demo was running (the last failed to start)
                name, checking = switcher.wait()
                continue
        print(f"\n=== {name} ({mode}) ===")
        server.reset_views(views)
        configure(switcher.settings)
        try:
            runner = demos[name](mode, server)
        except (Exception, SystemExit) as e:  # a robot not found, say: pick again from the page
            traceback.print_exc()
            switcher.failed(name, str(e) or type(e).__name__)
            print(f"{name} did not start; pick a demo on the dashboard (Ctrl+C ends).")
            name, checking = switcher.wait()
            continue
        switcher.started(name, runner)
        runner.run()  # returns once stopped: by the page (a switch, a check) or Ctrl+C
        name, checking = switcher.take()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default="real")
    parser.add_argument("--demo", choices=list(DEMOS), default="shooting_drill", help="the one to start with")
    parser.add_argument("--port", type=int, default=shooting_drill.DASHBOARD_PORT)
    args = parser.parse_args()
    server = attach_dashboard(port=args.port)
    switcher = demo_view.attach(server, demo_view.DemoSwitcher(list(DEMOS), SETTINGS, checks=args.mode == "real"))
    try:
        play(args.mode, args.demo, server, switcher)
    except KeyboardInterrupt:
        pass
    print("Stopped.")


if __name__ == "__main__":
    main()
