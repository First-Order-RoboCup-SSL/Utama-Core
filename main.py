from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.custom_referee import CustomReferee
from utama_core.entities.game.field import FieldBounds
from utama_core.kernel.kernel_strategy import build_give_and_go_solo_kernel_strategy
from utama_core.replay import ReplayWriterConfig
from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise
from utama_core.run import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


def main():
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=3, n_robots_blue=3, enable_gui=True)

    # Setup for real testing
    # Custom field size based setup in real
    # custom_bounds = FieldBounds(top_left=(-1.5, 1.125), bottom_right=(1.5, 1.125))

    runner = StrategyRunner(
        # Robot 0 is the goalkeeper (pinned outside the kernel scheduler), so
        # only robot 1 is an outfield tactic slot — a solo GiveAndGoTactic
        # pool, since PassAndShootTactic hard-requires 2 outfield robots.
        strategy=AbstractStrategy(build_kernel_strategy=build_give_and_go_solo_kernel_strategy((1,))),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=2,
        exp_enemy=0,
        replay_writer_config=ReplayWriterConfig(replay_name="test_replay", overwrite_existing=True),
        # field_bounds=custom_bounds,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        show_live_status=True,
        profiler_name=None,
        referee=referee,
    )
    runner.run()


if __name__ == "__main__":
    main()
