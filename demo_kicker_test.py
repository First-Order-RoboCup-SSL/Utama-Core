"""demo_kicker_test.py — Test the kicker with one robot on the Exhibition Road field.

Run:
    pixi run python demo_kicker_test.py

What this does
--------------
- 1 yellow robot, 0 opponents, Exhibition Road field (3 m × 2.25 m)
- KickerTestStrategy: robot fetches ball, aligns to face enemy goal, kicks,
  waits, then repeats
- Red goal center drawn in RSim
"""

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.kicker_test_strategy import KickerTestStrategy

N_FRIENDLY = 1
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True


def main() -> None:
    runner = StrategyRunner(
        strategy=KickerTestStrategy(robot_id=0),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="real",
        exp_friendly=N_FRIENDLY,
        exp_enemy=0,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        show_live_status=True,
        robot_id_map={1: 0},
    )

    runner.run()


if __name__ == "__main__":
    main()
