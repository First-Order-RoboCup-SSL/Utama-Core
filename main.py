from strategy.skills.go_to_ball import GoToBallStrategy
from run import StrategyRunner

if __name__ == "__main__":
    runner = StrategyRunner(
        strategy=GoToBallStrategy(target_id=0),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        exp_ball=True,
    )
    test = runner.run()
