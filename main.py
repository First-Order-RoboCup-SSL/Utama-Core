from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.strategies.defense_strategy import DefenceStrategy
from utama_core.strategy.examples.strategies.one_robot_placement_strategy import (
    RobotPlacementStrategy,
)
from utama_core.strategy.examples.strategies.startup_strategy import StartupStrategy

if __name__ == "__main__":
    # The robot we want to control
    target_robot_id = 0

    # Set up the runner
    runner = StrategyRunner(
        strategy=StartupStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
    )

    ScoreGoalStrategy(robot_id=target_robot_id).render()

    # Run the simulation
    test = runner.run()
