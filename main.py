from utama_core.replay import ReplayWriterConfig
from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import (
    DefenceStrategy,
    GoToBallExampleStrategy,
    RobotPlacementStrategy,
    StartupStrategy,
    RandomStrategy,
)
import argparse

def main():
    parser = argparse.ArgumentParser(description="Testing Seed")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--noisy_x", type=float, default=0.0, help="Random Seed")
    parser.add_argument("--noisy_y", type=float, default=0.0, help="Random Seed")
    args = parser.parse_args()
    runner = StrategyRunner(
        strategy=RandomStrategy(seed=args.seed),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        control_scheme="dwa",
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        print_real_fps=True,
        profiler_name=None,
        rsim_noise=RsimGaussianNoise(args.noisy_x, args.noisy_y)
    )
    runner.my_strategy.render()
    runner.run()


if __name__ == "__main__":
    main()
