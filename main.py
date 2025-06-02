from run import StrategyRunner
from strategy.skills.go_to_ball import GoToBallStrategy
from strategy.skills.solo_attacker import SoloAttackerStrategy
from strategy.skills.solo_defender import SoloDefenderStrategy

if __name__ == "__main__":
    runner = StrategyRunner(
        strategy=SoloAttackerStrategy(target_id=1),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="grsim",
        exp_friendly=3,
        exp_enemy=3,
        opp_strategy=SoloDefenderStrategy(target_id=0),
    )
    test = runner.run()
