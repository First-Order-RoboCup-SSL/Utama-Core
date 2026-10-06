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
        commands = []  # (sim time, command, stage) at each change, for the failure message
        for _ in range(60 * 60):
            runner.step_once()
            referee = runner.my.game.referee
            if referee is not None and (
                not commands or commands[-1][1:] != (referee.referee_command.name, referee.stage.name)
            ):
                commands.append(
                    (round(runner.my.current_game_frame.ts, 2), referee.referee_command.name, referee.stage.name)
                )
            if referee is not None and referee.stage == Stage.NORMAL_SECOND_HALF:
                kickoff = runner.my.current_game_frame
                break
        assert kickoff is not None, "the second half never started"

        assert runner.my_team_is_right is False
        assert (kickoff.my_team_is_right, runner.my.game.field.my_team_is_right) == (False, False)
        assert (runner.opp.current_game_frame.my_team_is_right, runner.opp.game.field.my_team_is_right) == (True, True)
        assert kickoff.referee.blue_team_on_positive_half is True
        seen = _describe(kickoff, commands)
        # The second half starts with its kick-off, taken by blue (yellow took the first).
        second_half = [c for _, c, stage in commands if stage == "NORMAL_SECOND_HALF_PRE"]
        assert "PREPARE_KICKOFF_BLUE" in second_half, seen
        assert commands[-1][1] == "NORMAL_START", seen
        # Each team in its own half, now the other one; the keepers were carried to their new
        # goals (driving there crossed both defense areas, and the fouls ended in a HALT).
        assert max(r.p.x for r in kickoff.friendly_robots.values()) < 0.1, seen
        assert min(r.p.x for r in kickoff.enemy_robots.values()) > -0.1, seen
        assert kickoff.friendly_robots[KEEPER].p.x < -GOAL_AREA_X, seen
        assert kickoff.enemy_robots[KEEPER].p.x > GOAL_AREA_X, seen
    finally:
        runner.close()


def _describe(frame, commands) -> str:
    def where(robots):
        return {i: (round(r.p.x, 2), round(r.p.y, 2)) for i, r in robots.items()}

    return f"t={frame.ts:.2f} friendly={where(frame.friendly_robots)} enemy={where(frame.enemy_robots)} commands={commands}"
