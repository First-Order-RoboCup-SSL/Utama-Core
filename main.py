from strategy.skills.go_to_ball import GoToBallStrategy
from strategy.examples.test.defense_test import DefendStrategy
from run import StrategyRunner

if __name__ == "__main__":
    # The robot we want to control
    target_robot_id = 0

    # Set up the runner
    runner = StrategyRunner(
        strategy=DefendStrategy(0),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=3,
        exp_enemy=2,
        opp_strategy=GoToBallStrategy(1,  opp_strategy=True)
    )

    # Run the simulation
    test = runner.run()

