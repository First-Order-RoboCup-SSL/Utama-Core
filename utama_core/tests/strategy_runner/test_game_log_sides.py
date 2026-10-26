"""The game log's referee records carry the sides: the converter reads them, our referee leaves them out."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.run.strategy_runner import StrategyRunner

DATA = RefereeData(
    source_identifier=None,
    time_sent=0.0,
    time_received=0.0,
    referee_command=RefereeCommand.HALT,
    referee_command_timestamp=0.0,
    stage=Stage.NORMAL_FIRST_HALF,
    stage_time_left=0.0,
    blue_team=TeamInfo("blue"),
    yellow_team=TeamInfo("yellow"),
)


@pytest.mark.parametrize(
    "yellow, right, blue_on_positive",
    [(True, True, False), (True, False, True), (False, True, True), (False, False, False)],
)
def test_sides_missing_from_the_referee_are_ours(yellow, right, blue_on_positive):
    runner = SimpleNamespace(my_team_is_yellow=yellow, my_team_is_right=right)
    assert StrategyRunner._with_sides(runner, DATA).blue_team_on_positive_half is blue_on_positive


def test_sides_the_referee_sent_are_kept():
    runner = SimpleNamespace(my_team_is_yellow=True, my_team_is_right=True)
    sent = dataclasses.replace(DATA, blue_team_on_positive_half=True)
    assert StrategyRunner._with_sides(runner, sent).blue_team_on_positive_half is True
