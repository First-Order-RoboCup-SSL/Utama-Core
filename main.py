from utama_core.entities.game import FieldBounds
from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import (
    DefenceStrategy,
    GoToBallExampleStrategy,
    RobotPlacementStrategy,
    StartupStrategy,
)


def main():
    runner = StrategyRunner(
        strategy=StartupStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="real",
        exp_friendly=1,
        exp_enemy=0,
        control_scheme="pid",
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        print_real_fps=True,
        profiler_name=None,
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    main()
