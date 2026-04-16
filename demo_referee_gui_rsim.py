"""demo_referee_gui_rsim.py — CustomReferee + web GUI + StrategyRunner (RSim).

Run:
    pixi run python demo_referee_gui_rsim.py
    # RSim window opens; open http://localhost:8080 in a browser

What it does:
  - Creates a CustomReferee (human profile) with enable_gui=True so the
    browser panel starts automatically.
  - Passes the referee to StrategyRunner via custom_referee= with
    referee_system="custom". StrategyRunner
    calls referee.step() on every tick and handles ball teleports on STOP
    automatically — no patching required.
  - WanderingStrategy is used as the base strategy so robots visibly move and
    you can watch the RefereeOverride tree interrupt them when you issue
    commands from the GUI (Halt, Kickoff Yellow, etc.).

Operator workflow:
  1. Open http://localhost:8080 in a browser.
  2. Robots start moving under WanderingStrategy.
  3. Click any command button (Halt, Stop, Kickoff Yellow…) — robots reposition.
  4. Click Normal Start to resume free play.
  5. With the human profile, the referee stays in STOP after a goal until the operator advances play.
"""

from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.run import StrategyRunner
from utama_core.tests.referee.wandering_strategy import WanderingStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFILE = "human"  # "human" or "simulation"
GUI_PORT = 8080
N_ROBOTS = 3  # robots per side
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    profile = load_profile(PROFILE)

    # enable_gui=True starts the HTTP server in a background daemon thread.
    # referee.step() is called by StrategyRunner on every tick; the GUI
    # receives state automatically after each call.
    referee = CustomReferee(
        profile,
        n_robots_yellow=N_ROBOTS,
        n_robots_blue=N_ROBOTS,
        enable_gui=True,
        gui_port=GUI_PORT,
    )

    runner = StrategyRunner(
        strategy=WanderingStrategy(),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        control_scheme="pid",
        exp_friendly=N_ROBOTS,
        exp_enemy=N_ROBOTS,
        referee_system="custom",
        custom_referee=referee,  # StrategyRunner drives referee.step() each tick
        show_live_status=True,
        opp_strategy=WanderingStrategy(),
    )

    runner.run()


if __name__ == "__main__":
    main()
