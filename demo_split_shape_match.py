"""Visible grsim 6v6: both teams run the split-shape kernel Strategy against each other.

Not a test — a manual/visual smoke run for reviewing LeadAndSupportTactic /
ShadowAndMarkTactic / the possession-split picker actually contesting an
opponent running the identical strategy, rendered by grSim itself (external
process — connect via the standard SSL ports; must already be running).
Runs for a fixed wall-clock duration rather than `runner.run()`'s
run-until-SIGINT, so it terminates on its own.

The referee GUI (enable_gui=True) shows a live per-robot debug panel via
`AbstractStrategy.debug_status()` — which tactic slot each robot is in, and
whether that slot is currently `committed()` (the "why won't this robot get
reassigned" signal in this model, since there's no BT node to point at).
Open http://localhost:8080 while this is running.
"""

import time

from utama_core.custom_referee import CustomReferee
from utama_core.kernel.kernel_strategy import build_split_shape_kernel_strategy
from utama_core.run import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy

DURATION_SECONDS = 90


def main():
    my_strategy = AbstractStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))
    opp_strategy = AbstractStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))

    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=6, n_robots_blue=6, enable_gui=True)

    runner = StrategyRunner(
        strategy=my_strategy,
        opp_strategy=opp_strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=6,
        exp_enemy=6,
        exp_ball=True,
        show_live_status=True,
        referee=referee,
    )

    start = time.monotonic()
    last_report = 0.0
    tick = 0
    try:
        while time.monotonic() - start < DURATION_SECONDS:
            runner.step_once()
            tick += 1
            elapsed = time.monotonic() - start
            if elapsed - last_report >= 5.0:
                last_report = elapsed
                ball = runner.my.game.ball.p
                r1 = runner.my.game.friendly_robots[1].p
                partition = my_strategy._kernel_strategy.active_partition
                shapes = {gid: sorted(robots) for gid, robots in partition.items()}
                print(
                    f"[t={elapsed:5.1f}s tick={tick}] ball=({ball.x:+.2f},{ball.y:+.2f}) "
                    f"my.robot1=({r1.x:+.2f},{r1.y:+.2f}) my.partition={shapes}",
                    flush=True,
                )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
