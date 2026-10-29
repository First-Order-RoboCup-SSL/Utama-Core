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

`planner_ab.py` and `calibrate_robot.py` are not here: they measure, with their own loop and
terminal prompts, rather than wait on the dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import signal
import traceback

import pass_receive
import save_and_counter
import score_goal
import shooting_drill

from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import demo as demo_view

# name -> build its runner on the shared dashboard: (mode, server) -> StrategyRunner
DEMOS = {
    "shooting_drill": lambda mode, server: shooting_drill.build_runner(mode, server=server),
    "pass_receive": lambda mode, server: pass_receive.build_runner(mode, server=server),
    "save_and_counter": lambda mode, server: save_and_counter.build_runner(mode, server=server),
    "score_goal": lambda mode, server: score_goal.build_runner(mode, server=server),
}


def play(mode: str, first: str, server, switcher: demo_view.DemoSwitcher, demos=DEMOS) -> None:
    """Plays `first`, then each demo the page asks for, until one is stopped otherwise (Ctrl+C)."""
    name = first
    while name is not None:
        signal.signal(signal.SIGINT, signal.default_int_handler)  # a runner sets its own while it runs
        print(f"\n=== {name} ({mode}) ===")
        try:
            runner = demos[name](mode, server)
        except (Exception, SystemExit) as e:  # a robot not found, say: pick again from the page
            traceback.print_exc()
            switcher.failed(name, str(e) or type(e).__name__)
            print(f"{name} did not start; pick a demo on the dashboard (Ctrl+C ends).")
            name = switcher.wait()
            continue
        switcher.started(name, runner)
        runner.run()  # returns once stopped: by the page (a switch) or Ctrl+C
        name = switcher.take()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default="real")
    parser.add_argument("--demo", choices=list(DEMOS), default="shooting_drill", help="the one to start with")
    parser.add_argument("--port", type=int, default=shooting_drill.DASHBOARD_PORT)
    args = parser.parse_args()
    server = attach_dashboard(port=args.port)
    switcher = demo_view.attach(server, demo_view.DemoSwitcher(list(DEMOS)))
    try:
        play(args.mode, args.demo, server, switcher)
    except KeyboardInterrupt:
        pass
    print("Stopped.")


if __name__ == "__main__":
    main()
