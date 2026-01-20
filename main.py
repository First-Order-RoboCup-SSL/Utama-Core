from utama_core.entities.game.field import FieldBounds
from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import (
    DefenceStrategy,
    GoToBallExampleStrategy,
    RobotPlacementStrategy,
    StartupStrategy,
    TwoRobotPlacementStrategy,
)


def main():
    # Define custom field bounds (e.g., a smaller region centered at 2, 2)
    custom_bounds = FieldBounds(top_left=(2.25, 1.5), bottom_right=(4.5, -1.5))

    runner = StrategyRunner(
        strategy=TwoRobotPlacementStrategy(first_robot_id=0, second_robot_id=1, field_bounds=custom_bounds),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
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
