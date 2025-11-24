from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy import DefenceStrategy, RobotPlacementStrategy, StartupStrategy


def main():
    runner = StrategyRunner(
        strategy=StartupStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=3,
        exp_enemy=3,
        control_scheme="fpp",
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        print_real_fps=True,
        profiler_name=None,
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
