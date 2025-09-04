from run import StrategyRunner
from strategy.examples.strategies.demo_strategy import DemoStrategy
from strategy.skills.score_goal import ScoreGoalStrategy

if __name__ == "__main__":
    # The robot we want to control
    target_robot_id = 0

    # Set up the runner
    runner = StrategyRunner(
        strategy=DemoStrategy(robot_id=0),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        opp_strategy=ScoreGoalStrategy(robot_id=0),
    )

    # Run the simulation
    test = runner.run()
