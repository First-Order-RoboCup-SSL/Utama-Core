"""demo_kicker_test.py — Test the kicker with one robot on the Exhibition Road field.

Run:
    pixi run python demo_kicker_test.py
    # Open http://localhost:8080 to watch

What this does
--------------
- 1 yellow robot, 0 opponents, Exhibition Road field (4 m × 3 m)
- KickerTestStrategy: robot approaches the ball, aligns toward an alternating
  target 1 m left/right of center, kicks, waits for the ball to settle, repeats
- Orange target point drawn in RSim on each kick leg
"""

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.custom_referee import CustomReferee
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.kicker_test_strategy import KickerTestStrategy

GUI_PORT = 8080
N_FRIENDLY = 1
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True


def main() -> None:
    referee = CustomReferee.from_profile_name(
        "simulation",
        n_robots_yellow=N_FRIENDLY,
        n_robots_blue=0,
        enable_gui=True,
        gui_port=GUI_PORT,
    )

    runner = StrategyRunner(
        strategy=KickerTestStrategy(robot_id=0),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        exp_friendly=N_FRIENDLY,
        exp_enemy=0,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        referee=referee,
        show_live_status=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
