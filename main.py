from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.strategies.demo_strategy import DemoStrategy
from utama_core.strategy.skills.score_goal import ScoreGoalStrategy

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
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
    )

    ScoreGoalStrategy(robot_id=target_robot_id).render()

    # Run the simulation
    test = runner.run()
