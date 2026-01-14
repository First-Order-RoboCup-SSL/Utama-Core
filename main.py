import atexit
import cProfile

from utama_core.motion_planning.src.common.control_schemes import ControlScheme
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
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        replay_writer_config=ReplayWriterConfig(replay_name="test-replay", overwrite_existing=True),
        control_scheme=ControlScheme.PID,
        print_real_fps=True,
        profiler_name=None,
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    main()
