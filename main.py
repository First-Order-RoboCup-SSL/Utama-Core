"""main.py — watch one match: two strategies play each other in rsim, refereed the way
round-robin matches are.

Run:
    pixi run main                          # split_shape vs high_press
    pixi run main tiki_taka counter_flow   # any two names from docs/strategies.md
    # the rsim window opens; the dashboard is at http://localhost:8080; Ctrl+C to stop

Yellow (the first strategy) starts on the right and kicks off. For many matches with
results, use `tools/tournament/round_robin.py`.
"""

import argparse

from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

N_ROBOTS = 6  # per side: robot 0 keeps goal, robots 1-5 play outfield
OUTFIELD_ROBOT_IDS = tuple(range(1, N_ROBOTS))


def _strategy(name: str) -> AbstractStrategy:
    build = getattr(kernel_strategy, f"build_{name}_kernel_strategy", None)
    if build is None:
        raise SystemExit(f"Unknown strategy {name!r}: see docs/strategies.md for the names")
    return AbstractStrategy(build_kernel_strategy=build(OUTFIELD_ROBOT_IDS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yellow", nargs="?", default="split_shape")
    parser.add_argument("blue", nargs="?", default="high_press")
    args = parser.parse_args()

    profile = load_profile("simulation")
    referee = CustomReferee(profile, n_robots_yellow=N_ROBOTS, n_robots_blue=N_ROBOTS)
    server = attach_dashboard()
    referee_view.attach(server, referee, profile)

    runner = StrategyRunner(
        strategy=_strategy(args.yellow),
        opp_strategy=_strategy(args.blue),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_ROBOTS,
        exp_enemy=N_ROBOTS,
        exp_ball=True,
        referee=referee,
        referee_initial_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
        show_live_status=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
