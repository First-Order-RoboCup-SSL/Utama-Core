"""demo_dribbler_test.py — Test the dribbler with one robot on the Exhibition Road field.

Run:
    pixi run python demo_dribbler_test.py
    # Open http://localhost:8080 to watch

What this does
--------------
- 1 yellow robot, 0 opponents, Exhibition Road field (4 m × 3 m)
- DribblerTestStrategy: robot fetches the ball, carries it with the dribbler
  to an alternating target left/right of center, then releases and repeats
- Blue target point drawn in RSim on each carry leg
"""

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.custom_referee import CustomReferee
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.dribbler_test_strategy import DribblerTestStrategy

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
        strategy=DribblerTestStrategy(robot_id=0),
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
