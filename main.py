from strategy.skills.go_to_ball import GoToBallStrategy
from run import StrategyRunner

if __name__ == "__main__":
    # The robot we want to control
    target_robot_id = 1

    # Set up the runner
    runner = StrategyRunner(
        strategy=GoToBallStrategy(target_robot_id),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
    )

    # Run the simulation
    test = runner.run()
