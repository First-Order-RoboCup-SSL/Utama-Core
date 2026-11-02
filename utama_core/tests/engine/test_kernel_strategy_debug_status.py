"""Tests for `AbstractStrategy.debug_status()` / `Strategy.slot_status()` — the
Tactic model's replacement for `_push_debug_status_to_referee`'s BT-node walk,
used to show "what is this robot's tactic doing right now" in the referee
GUI when there is no behaviour tree to walk.
"""

from __future__ import annotations

import pytest

from utama_core.engine.abstract_strategy import AbstractStrategy, robot_steps
from utama_core.engine.tactic import TacticTag
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.kernel_strategy import build_split_shape_kernel_strategy


@pytest.fixture
def split_shape_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = AbstractStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))
    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        exp_ball=True,
    )
    yield runner
    runner.close()


def test_debug_status_covers_goalkeeper_and_all_outfield_robots(split_shape_runner):
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    status = strategy.debug_status()

    assert status[0][:2] == ["goalkeeper", ""]
    for robot_id in (1, 2, 3, 4, 5):
        assert robot_id in status
        assert len(status[robot_id]) == 3  # [label, tag, steps]


def test_debug_status_labels_match_active_partition(split_shape_runner):
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    status = strategy.debug_status()
    partition = strategy._kernel_strategy.active_partition

    for tactic_id, robots in partition.items():
        for robot_id in robots:
            assert status[robot_id][0].startswith(tactic_id)


def test_slot_status_reports_committed_flag(split_shape_runner):
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    game = split_shape_runner.my.game
    status = strategy._kernel_strategy.slot_status(game)

    for tactic_id, info in status.items():
        assert isinstance(info["committed"], bool)
        assert info["robots"]


def test_debug_status_tag_is_the_assigned_tactics_own_tag(split_shape_runner):
    # The dashboard colours each robot by this second element, so it must be the tag the
    # tactic currently holding the robot declares, not a per-robot or per-strategy guess.
    split_shape_runner.step_once()
    strategy = split_shape_runner.my.strategy
    game = split_shape_runner.my.game
    status = strategy.debug_status()

    for tactic_id, info in strategy._kernel_strategy.slot_status(game).items():
        assert isinstance(info["tag"], TacticTag)
        for robot_id in info["robots"]:
            assert status[robot_id][1] == info["tag"].value


def test_debug_status_leaves_out_a_goalkeeper_whose_robot_is_not_on_the_field():
    """A test team without the goalkeeper's robot (e.g. one attacker alone) showed a
    phantom "goalkeeper" row for a robot ID that is not there."""
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = AbstractStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((0, 1)), goalkeeper_id=15)
    runner = StrategyRunner(
        strategy=strategy, my_team_is_yellow=True, my_team_is_right=True, mode="rsim", exp_friendly=2, exp_enemy=0
    )
    try:
        runner.step_once()
        status = runner.my.strategy.debug_status()
    finally:
        runner.close()
    assert 15 not in status
    assert set(status) == {0, 1}


def _refereed_runner(command, notifier=None):
    """`split_shape` under our own referee, seeded with `command`; `notifier` stands in for a dashboard."""
    from utama_core.custom_referee import CustomReferee
    from utama_core.run.strategy_runner import StrategyRunner

    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=6, n_robots_blue=3)
    if notifier is not None:
        referee.attach_dashboard_notifier(notifier)
    strategy = AbstractStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))
    return StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        exp_ball=True,
        referee=referee,
        referee_initial_command=command,
    )


@pytest.mark.parametrize(
    "command, label",
    [(RefereeCommand.STOP, "referee · stop · keeping clear"), (RefereeCommand.HALT, "halted")],
)
def test_a_robot_the_referee_drives_shows_its_job_not_its_tactic(command, label):
    """2026-10-28 field test: during a ball placement the Live tab still showed the robot's tactic,
    so the placer looked like it was going to the ball. Its slot keeps it through a restart."""
    runner = _refereed_runner(command)
    try:
        runner.step_once()
        status = runner.my.strategy.debug_status()
    finally:
        runner.close()
    for robot_id in (1, 2, 3, 4, 5):
        assert status[robot_id][:2] == [label, "referee"]


def test_in_live_play_no_robot_shows_the_referee():
    runner = _refereed_runner(RefereeCommand.FORCE_START)
    try:
        runner.step_once()
        status = runner.my.strategy.debug_status()
    finally:
        runner.close()
    assert all(row[1] != "referee" for row in status.values())


def test_the_placer_and_the_free_kick_taker_are_named():
    from utama_core.custom_referee.actions import (
        BallPlacementOursStep,
        DirectFreeOursStep,
    )
    from utama_core.engine.referee_override import _job

    placement = BallPlacementOursStep()
    placement._placer_id = 2
    assert _job(placement, 2) == "ball placement · placer"
    assert _job(placement, 3) == "ball placement · keeping off its path"
    free_kick = DirectFreeOursStep()
    free_kick._kicker_sticky.current = 4
    assert _job(free_kick, 4) == "our free kick · kicker"
    assert _job(free_kick, 1) == "our free kick · positions"


@pytest.mark.parametrize("watched", [True, False])
def test_a_watched_run_keeps_a_live_match_log_for_the_tactics_steps(watched):
    """Real runs have no match-log file, so no tactic traced what it was doing and the Live tab
    could not show it. With a dashboard attached the runner keeps one in memory."""
    runner = _refereed_runner(RefereeCommand.FORCE_START, notifier=(lambda *args: None) if watched else None)
    try:
        assert (runner.match_log is not None) is watched
        if watched:
            assert runner.my.strategy._kernel_strategy.match_log is runner.match_log
    finally:
        runner.close()


def test_a_robots_steps_are_the_plain_traces_naming_it():
    traces = {
        "shoot[3]": "turning to shoot",
        "go_to_ball[3].approach": "direct",
        "go_to_ball[13].approach": "shield",  # robot 13, not 3
        "give_and_go.shot_lane": {"open": True},  # not naming a robot
        "one_on_one[3].lane": {"x": 1.0},  # drawing data
    }
    assert robot_steps(traces, 3) == "shoot: turning to shoot · go_to_ball.approach: direct"
    assert robot_steps(traces, 5) == ""


def test_a_recorded_game_log_notes_each_robots_status_when_it_changes():
    """A field log said where each robot went but not what it was doing, so a placement read as
    a robot going to the ball there too."""
    runner = _refereed_runner(RefereeCommand.STOP)
    notes = []
    runner.game_log = type("Log", (), {"note": lambda self, text: notes.append(text)})()
    try:
        runner.step_once()
        runner.step_once()  # nothing changed: no new notes
    finally:
        runner.game_log = None
        runner.close()
    assert notes == [f"robot {rid}: referee · stop · keeping clear" for rid in range(6)]  # the keeper too
