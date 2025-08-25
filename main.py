from strategy.skills.go_to_ball import GoToBallStrategy
from strategy.test.defense_test import DefendStrategy
from strategy.skills.score_goal import ScoreGoalStrategy
from strategy.examples.strategies.demo_strategy import DemoStrategy
from strategy.skills.dribble import DribbleStrategy

from run import StrategyRunner

if __name__ == "__main__":
    # The robot we want to control
    target_robot_id = 0

    # Set up the runner
    runner = StrategyRunner(
        strategy=DemoStrategy(robot_id=0),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=3,
        exp_enemy=3,
        opp_strategy=DemoStrategy(robot_id=0),
    )

    # Run the simulation
    test = runner.run()
