"""main.py — watch one match: two strategies play each other in rsim, refereed the way
round-robin matches are.

Run:
    pixi run main                          # pendulum vs wolfpack
    pixi run main pendulum riptide   # any two names from docs/strategies.md
    pixi run main a.strategy wolfpack    # or a .strategy file (tools/dev/build_strategy.py)
    # the rsim window opens; the dashboard is at http://localhost:8080; Ctrl+C to stop

Yellow (the first strategy) starts on the right and kicks off. For many matches with
results, use `tools/evaluation/evaluate.py`.
"""

import argparse

from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.referee.profiles.profile_loader import load_profile
from utama_core.run.sim_setup import UnknownStrategy, load_side, sim_referee, sim_runner


def _strategy(name: str) -> AbstractStrategy:
    """A strategy by name, a `.strategy` file or `PATH.py:FUNCTION`; otherwise exit with the names there are."""
    try:
        return load_side(name)
    except UnknownStrategy as e:
        raise SystemExit(f"{e} (docs/strategies.md)") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yellow", nargs="?", default="pendulum")
    parser.add_argument("blue", nargs="?", default="wolfpack")
    args = parser.parse_args()
    yellow, blue = _strategy(args.yellow), _strategy(args.blue)  # fail on a bad name before the dashboard starts

    profile = load_profile("simulation")
    referee = sim_referee()
    server = attach_dashboard()
    referee_view.attach(server, referee, profile)

    runner, _ = sim_runner(yellow, blue, referee=referee, show_live_status=True)
    runner.run()


if __name__ == "__main__":
    main()
