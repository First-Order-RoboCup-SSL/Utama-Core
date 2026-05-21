"""demo_dribbler_test.py — Test the dribbler with one robot on the Exhibition Road field.

Run:
    pixi run python demo_dribbler_test.py

What this does
--------------
- 1 yellow robot, 0 opponents, Exhibition Road field (3 m × 2.25 m)
- DribblerTestStrategy: robot fetches the ball, carries it forward with the
  dribbler on, then stops
- Blue target point drawn in RSim during carry
"""

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.dribbler_test_strategy import DribblerTestStrategy

N_FRIENDLY = 1
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True


def main() -> None:
    runner = StrategyRunner(
        strategy=DribblerTestStrategy(robot_id=0),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        exp_friendly=N_FRIENDLY,
        exp_enemy=0,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        show_live_status=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
