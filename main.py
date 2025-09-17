from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.strategies.defense_strategy import DefenceStrategy

if __name__ == "__main__":
    # Set up the runner
    runner = StrategyRunner(
        strategy=DefenceStrategy(robot_id=0),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
    )

    runner.my_strategy.render()

    # Run the simulation
    test = runner.run()
