from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import (
    DefenceStrategy,
    GoToBallExampleStrategy,
    StartupStrategy,
    TwoRobotPlacementStrategy,
)


def main():
    runner = StrategyRunner(
        strategy=TwoRobotPlacementStrategy(robot_id=0),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="real",
        exp_friendly=2,
        exp_enemy=0,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        print_real_fps=True,
        profiler_name=None,
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    main()
