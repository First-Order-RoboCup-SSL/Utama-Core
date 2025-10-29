# profile_main.py
import atexit
import cProfile

from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.strategies.one_robot_placement_strategy import (
    RobotPlacementStrategy,
)
from utama_core.strategy.examples.strategies.startup_strategy import StartupStrategy

profiler = cProfile.Profile()
profiler.enable()


def dump():
    profiler.disable()
    profiler.dump_stats("sim_run.prof")


atexit.register(dump)


def main():
    runner = StrategyRunner(
        strategy=StartupStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
