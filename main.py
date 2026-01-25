from utama_core.entities.game import FieldBounds
from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import (
    GoToBallExampleStrategy,
    MultiTargetPlacementStrategy,
)


def main():
    # targets = [(4.129, -1.075), (2.668, -1.066), (4.029, 1.064), (2.519, 1.128)]
    custom_bounds = FieldBounds(top_left=(2.25, 1.5), bottom_right=(4.5, -1.5))

    # runner = StrategyRunner(
    #     strategy=MultiTargetPlacementStrategy(
    #         robot_id=1,
    #         targets=targets,
    #         reach_tolerance=0.05,
    #         loop_targets=True,
    #     ),
    #     my_team_is_yellow=True,
    #     my_team_is_right=True,
    #     mode="real",
    #     field_bounds=custom_bounds,
    #     control_scheme="fpp",
    #     exp_friendly=1,
    #     exp_enemy=5,
    #     replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
    #     print_real_fps=True,
    #     profiler_name=None,
    # )

    runner = StrategyRunner(
        strategy=GoToBallExampleStrategy(robot_id=1),
        # strategy=MultiTargetPlacementStrategy(
        #     robot_id=0,
        #     targets=targets,
        #     reach_tolerance=0.05,
        #     loop_targets=True,
        # ),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="real",
        field_bounds=custom_bounds,
        control_scheme="fpp",
        exp_friendly=1,
        exp_enemy=5,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        print_real_fps=True,
        profiler_name=None,
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    main()
