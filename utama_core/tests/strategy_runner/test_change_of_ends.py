"""At half-time the teams change ends (rulebook, Game Stages): `CustomReferee` swaps
`blue_team_on_positive_half` and `StrategyRunner` follows it, so both teams' frames, `Field`s
and kickoff positions are on the other side for the second half. Before, a match was played on
one side throughout and nothing could change it."""

from __future__ import annotations

import dataclasses

from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

N_OUTFIELD = 2
HALF_S = 3.0
KEEPER = 0
KEEPER_WALK_S = 15  # sim seconds for a keeper to cross the pitch to its new goal
GOAL_AREA_X = 3.5  # the goal line is at |x| = 4.5; the defense area is 1 m deep


def test_the_teams_play_the_second_half_from_the_other_end(headless):
    build = kernel_strategy.build_tiki_taka_kernel_strategy
    profile = load_profile("simulation")
    profile = dataclasses.replace(profile, game=dataclasses.replace(profile.game, half_duration_seconds=HALF_S))
    runner = StrategyRunner(
        strategy=AbstractStrategy(build_kernel_strategy=build(tuple(range(1, N_OUTFIELD + 1)))),
        opp_strategy=AbstractStrategy(build_kernel_strategy=build(tuple(range(1, N_OUTFIELD + 1)))),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=CustomReferee(profile, n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1),
        enable_vision_stream=False,
        referee_initial_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
    )
    try:
        assert runner.my.game.field.my_team_is_right is True
        kickoff = None
        for _ in range(60 * 60):
            runner.step_once()
            referee = runner.my.game.referee
            if referee is not None and referee.stage == Stage.NORMAL_SECOND_HALF:
                kickoff = runner.my.current_game_frame
                break
        assert kickoff is not None, "the second half never started"

        assert runner.my_team_is_right is False
        assert (kickoff.my_team_is_right, runner.my.game.field.my_team_is_right) == (False, False)
        assert (runner.opp.current_game_frame.my_team_is_right, runner.opp.game.field.my_team_is_right) == (True, True)
        assert kickoff.referee.blue_team_on_positive_half is True
        # Kickoff positions: each team's outfield robots in its own half, which is now the other one.
        assert max(r.p.x for i, r in kickoff.friendly_robots.items() if i != KEEPER) < 0.1
        assert min(r.p.x for i, r in kickoff.enemy_robots.items() if i != KEEPER) > -0.1
        # The keepers walk the length of the pitch, so they may still be on the way at the
        # kickoff (how far they got differs between CPUs); they reach their new goals.
        for _ in range(KEEPER_WALK_S * 60):
            runner.step_once()
            frame = runner.my.current_game_frame
            if frame.friendly_robots[KEEPER].p.x < -GOAL_AREA_X and frame.enemy_robots[KEEPER].p.x > GOAL_AREA_X:
                break
        assert frame.friendly_robots[KEEPER].p.x < -GOAL_AREA_X
        assert frame.enemy_robots[KEEPER].p.x > GOAL_AREA_X
    finally:
        runner.close()
