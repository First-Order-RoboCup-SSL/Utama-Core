from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import MultiTargetPlacementStrategy


def main():
    targets = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (-0.5, 0.5)]

    runner = StrategyRunner(
        strategy=MultiTargetPlacementStrategy(
            robot_id=0,
            targets=targets,
            reach_tolerance=0.05,
            loop_targets=True,
        ),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        print_real_fps=True,
        profiler_name=None,
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    main()
