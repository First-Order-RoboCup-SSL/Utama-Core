import atexit
import cProfile

from utama_core.replay import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.examples.strategies.defense_strategy import DefenceStrategy

profiler = cProfile.Profile()
profiler.enable()


def dump():
    profiler.disable()
    profiler.dump_stats("sim_run_pid.prof")


atexit.register(dump)

if __name__ == "__main__":
    # Set up the runner
    runner = StrategyRunner(
        strategy=DefenceStrategy(),
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
