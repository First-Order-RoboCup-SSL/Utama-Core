# profile_main.py
import atexit
import cProfile

from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy import DefenceStrategy, RobotPlacementStrategy, StartupStrategy

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
        mode="grsim",
        exp_friendly=6,
        exp_enemy=0,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        control_scheme="pid",
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
