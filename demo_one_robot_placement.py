"""demo_one_robot_placement.py — Entry point for testing RobotPlacementStrategy with one robot.

Run:
    pixi run python demo_one_robot_placement.py

What this sets up
-----------------
- Exhibition Road field (3 m × 2.25 m, ``GREAT_EXHIBITION_FIELD_DIMS``)
- 1 yellow robot, 0 opponents
- RobotPlacementStrategy: robot oscillates vertically around the field center
  while facing the ball
"""

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import RobotPlacementStrategy

N_FRIENDLY = 1
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True


def main() -> None:
    runner = StrategyRunner(
        strategy=RobotPlacementStrategy(robot_id=0),
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
