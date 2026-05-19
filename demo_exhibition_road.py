"""demo_exhibition_road.py — Exhibition Road Festival demo.

Run:
    pixi run python demo_exhibition_road.py
    # RSim window opens; open http://localhost:8080 in a browser

What it does:
  - Uses GREAT_EXHIBITION_FIELD_DIMS (4 m × 3 m) — the compact field sized
    for the Exhibition Road festival venue.
  - Builds a custom RefereeProfile inline (same style as Utama-Strategy/main.py)
    with all rules enabled and human-operator auto-advance settings so a
    festival steward can control the game from the browser GUI.
  - 2v2 format: two yellow robots vs two blue robots, matching the small field.
  - WanderingStrategy is used so robots visibly move and you can watch the
    RefereeOverride tree interrupt them when you issue commands from the GUI.
  - enable_gui=True starts the browser panel at http://localhost:8080 so the
    crowd can watch the referee state in real time.

Operator workflow:
  1. Open http://localhost:8080 in a browser.
  2. Robots begin under FORCE_START (rsim auto-seed).
  3. Click Halt / Stop / Kickoff Yellow… to intervene.
  4. Click Normal Start to resume.
  5. Goals are detected automatically; the referee waits for the operator to
     advance play (human auto-advance profile).
"""

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import (
    AutoAdvanceConfig,
    DefenseAreaConfig,
    GameConfig,
    GoalDetectionConfig,
    KeepOutConfig,
    OutOfBoundsConfig,
    RefereeProfile,
    RulesConfig,
)
from utama_core.run import StrategyRunner
from utama_core.tests.referee.wandering_strategy import WanderingStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GUI_PORT = 8080
N_ROBOTS = 2  # 2v2 fits the compact exhibition field
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True

# ---------------------------------------------------------------------------
# Exhibition Road referee profile
#
# Rules: all on, max 1 defender in area (2v2 — one is effectively GK),
#        keep-out radius scaled down to 0.3 m to match the smaller field.
# Game:  5-minute halves, human operator advances play after goals/restarts
#        so a steward can pause and commentate for the crowd.
# ---------------------------------------------------------------------------

_EXHIBITION_PROFILE = RefereeProfile(
    profile_name="exhibition_road",
    rules=RulesConfig(
        goal_detection=GoalDetectionConfig(
            enabled=True,
            cooldown_seconds=1.0,
        ),
        out_of_bounds=OutOfBoundsConfig(
            enabled=True,
            free_kick_assigner="last_touch",
        ),
        defense_area=DefenseAreaConfig(
            enabled=False,
            max_defenders=1,  # one robot allowed in own penalty area
            attacker_infringement=True,
        ),
        keep_out=KeepOutConfig(
            enabled=False,
            radius_meters=0.3,  # scaled down from standard 0.5 m for small field
            violation_persistence_frames=30,
        ),
    ),
    game=GameConfig(
        half_duration_seconds=300.0,  # 5-minute halves
        kickoff_team="yellow",
        force_start_after_goal=False,  # operator manually advances after goals
        stop_duration_seconds=3.0,
        prepare_duration_seconds=3.0,
        kickoff_timeout_seconds=10.0,
        auto_advance=AutoAdvanceConfig(
            # Human operator controls all state transitions — prevents robots
            # from suddenly moving while the crowd is close to the field.
            stop_to_next_command=False,
            prepare_kickoff_to_normal=False,
            prepare_penalty_to_normal=False,
            direct_free_to_normal=False,
            ball_placement_to_next=False,
            normal_start_to_force=True,  # still auto-force if kickoff stalls
        ),
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    referee = CustomReferee(
        _EXHIBITION_PROFILE,
        n_robots_yellow=N_ROBOTS,
        n_robots_blue=N_ROBOTS,
        enable_gui=True,
        gui_port=GUI_PORT,
    )

    runner = StrategyRunner(
        strategy=WanderingStrategy(field_dims=GREAT_EXHIBITION_FIELD_DIMS),
        opp_strategy=WanderingStrategy(field_dims=GREAT_EXHIBITION_FIELD_DIMS),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        control_scheme="pid",
        exp_friendly=N_ROBOTS,
        exp_enemy=N_ROBOTS,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,  # 4 m × 3 m exhibition field
        referee=referee,
        show_live_status=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
