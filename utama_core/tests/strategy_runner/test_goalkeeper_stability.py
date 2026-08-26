"""GoalkeeperTactic motion-stability regression test.

Found live during the gap #6/#9 referee-rule tournament validation
(2026-08-26, `replays/gap6_validation_20260826_124653/`): the goalkeeper
oscillated with a ~1.6s period and ~0.74m amplitude around a completely
static target (ball at rest during `PREPARE_KICKOFF`, predicted goal-line
intercept fixed with <1e-6 drift) — never converging. Isolated `PIDController`
gains (same gains, same start/target, no `StrategyRunner`/rsim in the loop)
converged cleanly, so the instability only reproduced through the full sim
loop with `ctx.motion_controller` — which for a `StrategyRunner` defaults to
`"fpp"` (`FastPathPlanningController`), the same scheme every outfield tactic
uses for obstacle-avoiding motion. The goalkeeper's task never needs
obstacle-avoidance path planning (it holds a point on its own goal line
inside its own defense area, where no legal opponent/teammate should be
routing through), so `GoalkeeperTactic` (`utama_core/tactics/goalkeeper.py`)
now builds and uses its own dedicated `PIDController` instead of sharing
`ctx.motion_controller` — verified below to eliminate the oscillation
without needing to debug `FastPathPlanner`'s detour logic directly.
"""

from __future__ import annotations

import dataclasses

from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

N_OUTFIELD = 5
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))

# Same pairing that surfaced the bug live.
_BUILD_A = kernel_strategy.build_counter_flow_kernel_strategy
_BUILD_B = kernel_strategy.build_tiki_taka_kernel_strategy


def test_goalkeeper_converges_on_static_target_without_oscillating(headless):
    """Ball parked at kickoff (never advances past PREPARE_KICKOFF, so the
    keeper's target is provably fixed) — the keeper must settle to a steady
    y-position, not oscillate around it.
    """
    strategy_a = AbstractStrategy(build_kernel_strategy=_BUILD_A(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=_BUILD_B(OUTFIELD_ROBOT_IDS))

    profile = load_profile("simulation")
    profile = dataclasses.replace(
        profile,
        game=dataclasses.replace(
            profile.game,
            kickoff_team="blue",
            prepare_duration_seconds=9999.0,  # never auto-advance to NORMAL_START
        ),
    )
    referee = CustomReferee(profile, n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1)

    runner = StrategyRunner(
        strategy=strategy_a,
        opp_strategy=strategy_b,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=referee,
        enable_vision_stream=False,
        referee_initial_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
    )
    try:
        ys = []
        for _ in range(int(7.5 * 60)):
            runner.step_once()
            gk = runner.my.current_game_frame.friendly_robots.get(0)
            if gk is not None:
                ys.append(gk.p.y)
    finally:
        runner.close()

    # Pre-fix (ctx.motion_controller / "fpp") measured 0.385m in this same
    # window; post-fix (dedicated PIDController) measured 0.000m.
    steady_state = ys[-120:]  # last 2s
    steady_range = max(steady_state) - min(steady_state)
    assert steady_range < 0.05, f"goalkeeper is oscillating in steady state: {steady_range:.4f}m range"
