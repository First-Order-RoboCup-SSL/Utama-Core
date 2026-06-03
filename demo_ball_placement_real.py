"""demo_ball_placement_real.py — Test ball placement with a single real robot.

Run:
    pixi run python demo_ball_placement_real.py
    # Then open http://localhost:8080 in a browser

What this sets up
-----------------
- Exhibition Road field (4 m x 3 m)
- 1 yellow robot, no opponents
- CustomReferee starts in HALT — the robot will not move until you issue a
  command from the GUI
- All automatic rule detection (out-of-bounds, goal, keep-out) is disabled so
  the only commands come from the operator via the GUI

Workflow
--------
1. Run this script and open http://localhost:8080.
2. Place the ball somewhere on the field by hand.
3. In the GUI "Manual Commands" panel, set a target position and issue
   BALL_PLACEMENT_YELLOW.
4. The robot drives to the ball with its dribbler on and attempts to carry it
   to the target.
5. Issue HALT from the GUI at any time to stop the robot immediately.

Safety notes
------------
- The robot starts in HALT and will not move until you issue a command.
- Keep the target position within the physical field bounds.
- Issue HALT from the GUI before approaching the field.
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

# ---------------------------------------------------------------------------
# Configuration -- edit these to match your setup
# ---------------------------------------------------------------------------

GUI_PORT = 8080
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True

YELLOW_VISION_TO_CMD: dict[int, int] = {
    1: 3,  # vision 1 → firmware 3
}

# ---------------------------------------------------------------------------
# Referee profile
#
# All automatic rules are OFF. Commands come from the operator only.
# auto_advance is also OFF so the state never transitions without you.
# ---------------------------------------------------------------------------

_REAL_TEST_PROFILE = RefereeProfile(
    profile_name="ball_placement_real_test",
    rules=RulesConfig(
        goal_detection=GoalDetectionConfig(
            enabled=False,
            cooldown_seconds=1.0,
        ),
        out_of_bounds=OutOfBoundsConfig(
            enabled=False,
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
            stop_to_next_command=False,
            prepare_kickoff_to_normal=False,
            prepare_penalty_to_normal=False,
            direct_free_to_normal=False,
            ball_placement_to_next=False,
            normal_start_to_force=False,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    referee = CustomReferee(
        _REAL_TEST_PROFILE,
        n_robots_yellow=1,
        n_robots_blue=0,
        enable_gui=True,
        gui_port=GUI_PORT,
    )

    runner = StrategyRunner(
        strategy=BallPlacementStrategy(),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="real",
        exp_friendly=1,
        exp_enemy=0,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        referee=referee,
        yellow_vision_to_cmd_mapping=YELLOW_VISION_TO_CMD,
        show_live_status=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
