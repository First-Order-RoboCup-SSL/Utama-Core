"""demo_ball_placement.py — Entry point for the ball placement feature.

Run:
    pixi run python demo_ball_placement.py
    # RSim window opens; open http://localhost:8080 in a browser

What this sets up
-----------------
- Exhibition Road field (4 m × 3 m, ``GREAT_EXHIBITION_FIELD_DIMS``)
- 2v2 format: two yellow robots (your team) vs two blue robots (wandering
  opponents that trigger out-of-bounds events naturally)
- CustomReferee pre-configured with:
    - Goal detection enabled (issues PREPARE_KICKOFF → NORMAL_START cycle)
    - Out-of-bounds enabled (issues STOP → BALL_PLACEMENT_YELLOW → DIRECT_FREE)
    - Defense area and keep-out rules OFF (less noise during development)
    - Auto-advance fully enabled so state transitions fire automatically —
      you can watch the full placement → free-kick → restart cycle without
      touching the GUI
- Browser GUI at http://localhost:8080 — use "Manual Commands" to fire
  BALL_PLACEMENT_YELLOW at any time and set the target position

Your task
---------
Open ``utama_core/strategy/examples/ball_placement_strategy.py`` and implement
``BallPlacementStep.update()``.  Everything else here is already wired up.

Workflow
--------
1. Run this script and open http://localhost:8080.
2. Kick the ball out of bounds (it happens naturally with wandering robots) or
   use the GUI "Manual Commands" panel to issue BALL_PLACEMENT_YELLOW.
3. Watch one of your robots (yellow) drive to the ball, capture it with the
   dribbler, and carry it to the target circle shown in the GUI.
4. Iterate until the test suite passes:
       pixi run pytest utama_core/tests/strategy_runner/test_ball_placement_rsim.py -v
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
from utama_core.strategy.examples.ball_placement_strategy import BallPlacementStrategy
from utama_core.tests.referee.wandering_strategy import WanderingStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GUI_PORT = 8080
N_ROBOTS = 2
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True

# ---------------------------------------------------------------------------
# Referee profile
#
# Out-of-bounds is the primary trigger for ball placement in a real game.
# Defense area and keep-out rules are disabled to reduce noise while you're
# developing the placement skill itself.
# ---------------------------------------------------------------------------

_BALL_PLACEMENT_PROFILE = RefereeProfile(
    profile_name="ball_placement_dev",
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
            max_defenders=1,
            attacker_infringement=False,
        ),
        keep_out=KeepOutConfig(
            enabled=False,
            radius_meters=0.3,
            violation_persistence_frames=30,
        ),
    ),
    game=GameConfig(
        half_duration_seconds=300.0,
        kickoff_team="yellow",
        force_start_after_goal=False,
        stop_duration_seconds=2.0,
        prepare_duration_seconds=3.0,
        kickoff_timeout_seconds=10.0,
        auto_advance=AutoAdvanceConfig(
            # All auto-advance enabled: state machine drives itself so you can
            # observe the full placement → free-kick → normal-start cycle.
            stop_to_next_command=True,
            prepare_kickoff_to_normal=True,
            prepare_penalty_to_normal=True,
            direct_free_to_normal=True,
            ball_placement_to_next=True,
            normal_start_to_force=True,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    referee = CustomReferee(
        _BALL_PLACEMENT_PROFILE,
        n_robots_yellow=N_ROBOTS,
        n_robots_blue=N_ROBOTS,
        enable_gui=True,
        gui_port=GUI_PORT,
    )

    runner = StrategyRunner(
        strategy=BallPlacementStrategy(),
        # Opponents wander so they occasionally kick the ball out of bounds,
        # naturally triggering ball placement without manual intervention.
        opp_strategy=WanderingStrategy(field_dims=GREAT_EXHIBITION_FIELD_DIMS),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        control_scheme="pid",
        exp_friendly=N_ROBOTS,
        exp_enemy=N_ROBOTS,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        referee=referee,
        show_live_status=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
