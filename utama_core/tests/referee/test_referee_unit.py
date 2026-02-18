"""Unit tests for the referee integration layer.

Tests cover:
  - RefereeData new fields (game_events, match_type, status_message) and custom __eq__
  - RefereeRefiner.refine injects data into GameFrame; deduplication logic
  - Game.referee property proxies correctly from CurrentGameFrame
  - CheckRefereeCommand condition node (SUCCESS / FAILURE / None-referee guard)
  - Dispatcher routing (ours vs. theirs) for bilateral commands
  - build_referee_override_tree structure and priority
"""

from types import SimpleNamespace

import py_trees
import pytest

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field
from utama_core.entities.game.game import Game
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import GameHistory
from utama_core.entities.game.robot import Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.run.refiners.referee import RefereeRefiner
from utama_core.strategy.referee.conditions import CheckRefereeCommand
from utama_core.strategy.referee.tree import build_referee_override_tree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _team_info(goalkeeper: int = 0) -> TeamInfo:
    return TeamInfo(
        name="TestTeam",
        score=0,
        red_cards=0,
        yellow_card_times=[],
        yellow_cards=0,
        timeouts=0,
        timeout_time=0,
        goalkeeper=goalkeeper,
    )


def _make_referee_data(
    command: RefereeCommand = RefereeCommand.HALT,
    stage: Stage = Stage.NORMAL_FIRST_HALF,
    game_events=None,
    match_type: int = 0,
    status_message=None,
) -> RefereeData:
    return RefereeData(
        source_identifier="test",
        time_sent=1.0,
        time_received=1.0,
        referee_command=command,
        referee_command_timestamp=1.0,
        stage=stage,
        stage_time_left=300.0,
        blue_team=_team_info(goalkeeper=1),
        yellow_team=_team_info(goalkeeper=2),
        game_events=game_events if game_events is not None else [],
        match_type=match_type,
        status_message=status_message,
    )


def _robot(robot_id: int, x: float = 0.0, y: float = 0.0) -> Robot:
    zv = Vector2D(0, 0)
    return Robot(id=robot_id, is_friendly=True, has_ball=False, p=Vector2D(x, y), v=zv, a=zv, orientation=0.0)


def _ball(x: float = 0.0, y: float = 0.0) -> Ball:
    zv = Vector3D(0, 0, 0)
    return Ball(p=Vector3D(x, y, 0), v=zv, a=zv)


def _make_game_frame(
    friendly_robots=None,
    referee=None,
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = True,
) -> GameFrame:
    if friendly_robots is None:
        friendly_robots = {0: _robot(0)}
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly_robots,
        enemy_robots={},
        ball=_ball(),
        referee=referee,
    )


def _make_game(
    friendly_robots=None,
    referee=None,
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = True,
) -> Game:
    frame = _make_game_frame(friendly_robots, referee, my_team_is_yellow, my_team_is_right)
    history = GameHistory()
    return Game(past=history, current=frame, field=Field.FULL_FIELD_BOUNDS)


def _make_blackboard(game: Game, cmd_map=None):
    """Construct a minimal SimpleNamespace blackboard as used by AbstractBehaviour."""
    bb = SimpleNamespace()
    bb.game = game
    bb.cmd_map = cmd_map if cmd_map is not None else {}
    bb.motion_controller = None
    return bb


# ---------------------------------------------------------------------------
# RefereeData — new fields and __eq__
# ---------------------------------------------------------------------------


class TestRefereeDataNewFields:
    def test_default_game_events_is_empty_list(self):
        data = _make_referee_data()
        assert data.game_events == []

    def test_default_match_type_is_zero(self):
        data = _make_referee_data()
        assert data.match_type == 0

    def test_default_status_message_is_none(self):
        data = _make_referee_data()
        assert data.status_message is None

    def test_custom_game_events_stored(self):
        events = [object(), object()]
        data = _make_referee_data(game_events=events)
        assert data.game_events is events

    def test_custom_match_type_stored(self):
        data = _make_referee_data(match_type=2)
        assert data.match_type == 2

    def test_custom_status_message_stored(self):
        data = _make_referee_data(status_message="Foul by blue")
        assert data.status_message == "Foul by blue"

    def test_eq_ignores_game_events(self):
        """Two records with different game_events but identical core fields must compare equal."""
        a = _make_referee_data(game_events=[])
        b = _make_referee_data(game_events=["something"])
        assert a == b

    def test_eq_ignores_match_type(self):
        a = _make_referee_data(match_type=0)
        b = _make_referee_data(match_type=3)
        assert a == b

    def test_eq_ignores_status_message(self):
        a = _make_referee_data(status_message=None)
        b = _make_referee_data(status_message="Ball out of bounds")
        assert a == b

    def test_eq_sensitive_to_referee_command(self):
        a = _make_referee_data(command=RefereeCommand.HALT)
        b = _make_referee_data(command=RefereeCommand.STOP)
        assert a != b

    def test_eq_sensitive_to_stage(self):
        a = _make_referee_data(stage=Stage.NORMAL_FIRST_HALF)
        b = _make_referee_data(stage=Stage.NORMAL_SECOND_HALF)
        assert a != b


# ---------------------------------------------------------------------------
# RefereeRefiner
# ---------------------------------------------------------------------------


class TestRefereeRefiner:
    def setup_method(self):
        self.refiner = RefereeRefiner()

    def test_refine_none_data_returns_original_frame(self):
        frame = _make_game_frame()
        result = self.refiner.refine(frame, None)
        assert result is frame

    def test_refine_injects_referee_into_frame(self):
        frame = _make_game_frame(referee=None)
        data = _make_referee_data(command=RefereeCommand.STOP)
        result = self.refiner.refine(frame, data)
        assert result.referee is data

    def test_refine_preserves_other_frame_fields(self):
        robots = {0: _robot(0, 1.0, 2.0)}
        frame = _make_game_frame(friendly_robots=robots, my_team_is_yellow=False)
        data = _make_referee_data()
        result = self.refiner.refine(frame, data)
        assert result.my_team_is_yellow is False
        assert result.friendly_robots == robots

    def test_first_data_is_always_recorded(self):
        data = _make_referee_data()
        frame = _make_game_frame()
        self.refiner.refine(frame, data)
        assert len(self.refiner._referee_records) == 1

    def test_duplicate_data_not_re_recorded(self):
        """Records with same core fields (equal by __eq__) are not duplicated."""
        data1 = _make_referee_data()
        data2 = _make_referee_data(status_message="different but equal core")
        frame = _make_game_frame()
        self.refiner.refine(frame, data1)
        self.refiner.refine(frame, data2)
        assert len(self.refiner._referee_records) == 1

    def test_changed_command_is_recorded(self):
        frame = _make_game_frame()
        data1 = _make_referee_data(command=RefereeCommand.HALT)
        data2 = _make_referee_data(command=RefereeCommand.STOP)
        self.refiner.refine(frame, data1)
        self.refiner.refine(frame, data2)
        assert len(self.refiner._referee_records) == 2

    def test_last_command_property(self):
        frame = _make_game_frame()
        self.refiner.refine(frame, _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_BLUE))
        assert self.refiner.last_command == RefereeCommand.BALL_PLACEMENT_BLUE

    def test_last_command_defaults_to_halt_when_empty(self):
        assert self.refiner.last_command == RefereeCommand.HALT

    def test_source_identifier_none_when_empty(self):
        assert self.refiner.source_identifier() is None

    def test_source_identifier_after_record(self):
        frame = _make_game_frame()
        self.refiner.refine(frame, _make_referee_data())
        assert self.refiner.source_identifier() == "test"


# ---------------------------------------------------------------------------
# Game.referee property
# ---------------------------------------------------------------------------


class TestGameRefereeProperty:
    def test_referee_none_when_no_data(self):
        game = _make_game(referee=None)
        assert game.referee is None

    def test_referee_returns_injected_data(self):
        data = _make_referee_data(command=RefereeCommand.STOP)
        game = _make_game(referee=data)
        assert game.referee is data

    def test_referee_command_accessible(self):
        data = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_YELLOW)
        game = _make_game(referee=data)
        assert game.referee.referee_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    def test_add_game_frame_updates_referee(self):
        """After add_game_frame, game.referee reflects the new frame."""
        game = _make_game(referee=None)
        assert game.referee is None

        new_data = _make_referee_data(command=RefereeCommand.HALT)
        new_frame = _make_game_frame(referee=new_data)
        game.add_game_frame(new_frame)
        assert game.referee is new_data


# ---------------------------------------------------------------------------
# CheckRefereeCommand condition node
# ---------------------------------------------------------------------------


def _setup_check_node(*commands: RefereeCommand, game: Game) -> CheckRefereeCommand:
    """Build and set up a CheckRefereeCommand node with the given expected commands."""
    # Reset py_trees blackboard between tests
    py_trees.blackboard.Blackboard.enable_activity_stream()
    node = CheckRefereeCommand(*commands)
    node.blackboard = _make_blackboard(game)
    return node


class TestCheckRefereeCommand:
    def test_returns_failure_when_referee_is_none(self):
        game = _make_game(referee=None)
        node = CheckRefereeCommand(RefereeCommand.HALT)
        node.blackboard = _make_blackboard(game)
        assert node.update() == py_trees.common.Status.FAILURE

    def test_returns_success_on_matching_single_command(self):
        data = _make_referee_data(command=RefereeCommand.HALT)
        game = _make_game(referee=data)
        node = CheckRefereeCommand(RefereeCommand.HALT)
        node.blackboard = _make_blackboard(game)
        assert node.update() == py_trees.common.Status.SUCCESS

    def test_returns_failure_on_non_matching_command(self):
        data = _make_referee_data(command=RefereeCommand.STOP)
        game = _make_game(referee=data)
        node = CheckRefereeCommand(RefereeCommand.HALT)
        node.blackboard = _make_blackboard(game)
        assert node.update() == py_trees.common.Status.FAILURE

    def test_returns_success_on_any_matching_multi_command(self):
        for cmd in (RefereeCommand.TIMEOUT_YELLOW, RefereeCommand.TIMEOUT_BLUE):
            data = _make_referee_data(command=cmd)
            game = _make_game(referee=data)
            node = CheckRefereeCommand(RefereeCommand.TIMEOUT_YELLOW, RefereeCommand.TIMEOUT_BLUE)
            node.blackboard = _make_blackboard(game)
            assert node.update() == py_trees.common.Status.SUCCESS

    def test_returns_failure_when_command_not_in_multi_list(self):
        data = _make_referee_data(command=RefereeCommand.HALT)
        game = _make_game(referee=data)
        node = CheckRefereeCommand(RefereeCommand.TIMEOUT_YELLOW, RefereeCommand.TIMEOUT_BLUE)
        node.blackboard = _make_blackboard(game)
        assert node.update() == py_trees.common.Status.FAILURE

    def test_node_name_contains_command_names(self):
        node = CheckRefereeCommand(RefereeCommand.HALT, RefereeCommand.STOP)
        assert "HALT" in node.name
        assert "STOP" in node.name


# ---------------------------------------------------------------------------
# HaltStep and StopStep — basic output verification
# ---------------------------------------------------------------------------


def _make_cmd_map(game: Game) -> dict:
    return {rid: None for rid in game.friendly_robots}


class TestHaltAndStopStep:
    def _run_step(self, step_class, game: Game) -> tuple:
        from types import SimpleNamespace

        cmd_map = _make_cmd_map(game)
        bb = _make_blackboard(game, cmd_map)
        node = step_class(name="TestStep")
        node.blackboard = bb
        status = node.update()
        return status, cmd_map

    def test_halt_returns_running(self):
        from utama_core.strategy.referee.actions import HaltStep

        game = _make_game(referee=_make_referee_data(command=RefereeCommand.HALT))
        status, _ = self._run_step(HaltStep, game)
        assert status == py_trees.common.Status.RUNNING

    def test_halt_writes_to_all_robots(self):
        from utama_core.strategy.referee.actions import HaltStep

        robots = {0: _robot(0), 1: _robot(1)}
        game = _make_game(friendly_robots=robots, referee=_make_referee_data())
        status, cmd_map = self._run_step(HaltStep, game)
        assert set(cmd_map.keys()) == {0, 1}
        for rid in robots:
            assert cmd_map[rid] is not None

    def test_stop_returns_running(self):
        from utama_core.strategy.referee.actions import StopStep

        game = _make_game(referee=_make_referee_data(command=RefereeCommand.STOP))
        status, _ = self._run_step(StopStep, game)
        assert status == py_trees.common.Status.RUNNING

    def test_stop_writes_to_all_robots(self):
        from utama_core.strategy.referee.actions import StopStep

        robots = {0: _robot(0), 1: _robot(1), 2: _robot(2)}
        game = _make_game(friendly_robots=robots, referee=_make_referee_data())
        status, cmd_map = self._run_step(StopStep, game)
        assert set(cmd_map.keys()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# build_referee_override_tree — structure checks
# ---------------------------------------------------------------------------


class TestRefereeOverrideTreeStructure:
    def setup_method(self):
        self.tree = build_referee_override_tree()

    def test_root_is_selector(self):
        assert isinstance(self.tree, py_trees.composites.Selector)

    def test_root_name(self):
        assert self.tree.name == "RefereeOverride"

    def test_has_eleven_children(self):
        # HALT, STOP, TIMEOUT, BALL_PLACEMENT×2, KICKOFF×2, PENALTY×2, DIRECT_FREE×2
        assert len(self.tree.children) == 11

    def test_each_child_is_sequence(self):
        for child in self.tree.children:
            assert isinstance(child, py_trees.composites.Sequence)

    def test_each_sequence_has_two_children(self):
        for child in self.tree.children:
            assert len(child.children) == 2

    def test_each_sequence_first_child_is_check_command(self):
        for child in self.tree.children:
            assert isinstance(child.children[0], CheckRefereeCommand)

    def test_halt_is_first(self):
        first_seq = self.tree.children[0]
        condition = first_seq.children[0]
        assert RefereeCommand.HALT in condition.expected_commands

    def test_stop_is_second(self):
        second_seq = self.tree.children[1]
        condition = second_seq.children[0]
        assert RefereeCommand.STOP in condition.expected_commands

    def test_timeout_handles_both_colours(self):
        timeout_seq = self.tree.children[2]
        condition = timeout_seq.children[0]
        assert RefereeCommand.TIMEOUT_YELLOW in condition.expected_commands
        assert RefereeCommand.TIMEOUT_BLUE in condition.expected_commands

    def test_all_bilateral_commands_covered(self):
        """Every bilateral referee command must appear in at least one condition node."""
        covered = set()
        for child in self.tree.children:
            condition = child.children[0]
            covered.update(condition.expected_commands)

        bilateral = {
            RefereeCommand.BALL_PLACEMENT_YELLOW,
            RefereeCommand.BALL_PLACEMENT_BLUE,
            RefereeCommand.PREPARE_KICKOFF_YELLOW,
            RefereeCommand.PREPARE_KICKOFF_BLUE,
            RefereeCommand.PREPARE_PENALTY_YELLOW,
            RefereeCommand.PREPARE_PENALTY_BLUE,
            RefereeCommand.DIRECT_FREE_YELLOW,
            RefereeCommand.DIRECT_FREE_BLUE,
        }
        assert bilateral.issubset(covered)


# ---------------------------------------------------------------------------
# Dispatcher ours-vs-theirs routing (no actual motion controller required)
# ---------------------------------------------------------------------------


def _make_dispatch_blackboard(game: Game) -> SimpleNamespace:
    bb = _make_blackboard(game)
    return bb


class TestDispatcherRouting:
    """Verify that dispatcher nodes call the correct Ours/Theirs child based on team colour."""

    def _tick_dispatcher(self, dispatcher, game: Game) -> py_trees.common.Status:
        cmd_map = {rid: None for rid in game.friendly_robots}
        bb = _make_blackboard(game, cmd_map)
        dispatcher.blackboard = bb
        # Propagate blackboard to inner ours/theirs nodes
        dispatcher._ours.blackboard = bb
        dispatcher._theirs.blackboard = bb
        return dispatcher.update()

    def test_ball_placement_yellow_calls_ours_when_yellow(self):
        from utama_core.strategy.referee.actions import (
            BallPlacementOursStep,
            BallPlacementTheirsStep,
        )
        from utama_core.strategy.referee.tree import _BallPlacementDispatch

        data = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        # my_team_is_yellow=True, is_yellow_command=True → ours
        game = _make_game(referee=data, my_team_is_yellow=True)
        dispatcher = _BallPlacementDispatch(is_yellow_command=True, name="test")

        called = []
        # original_ours = dispatcher._ours.update
        # original_theirs = dispatcher._theirs.update
        dispatcher._ours.update = lambda: called.append("ours") or py_trees.common.Status.RUNNING
        dispatcher._theirs.update = lambda: called.append("theirs") or py_trees.common.Status.RUNNING

        self._tick_dispatcher(dispatcher, game)
        assert called == ["ours"]

    def test_ball_placement_yellow_calls_theirs_when_blue(self):
        from utama_core.strategy.referee.tree import _BallPlacementDispatch

        data = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        # my_team_is_yellow=False, is_yellow_command=True → theirs
        game = _make_game(referee=data, my_team_is_yellow=False)
        dispatcher = _BallPlacementDispatch(is_yellow_command=True, name="test")

        called = []
        dispatcher._ours.update = lambda: called.append("ours") or py_trees.common.Status.RUNNING
        dispatcher._theirs.update = lambda: called.append("theirs") or py_trees.common.Status.RUNNING

        self._tick_dispatcher(dispatcher, game)
        assert called == ["theirs"]

    def test_kickoff_blue_calls_ours_when_blue(self):
        from utama_core.strategy.referee.tree import _KickoffDispatch

        data = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_BLUE)
        # my_team_is_yellow=False, is_yellow_command=False → ours
        game = _make_game(referee=data, my_team_is_yellow=False)
        dispatcher = _KickoffDispatch(is_yellow_command=False, name="test")

        called = []
        dispatcher._ours.update = lambda: called.append("ours") or py_trees.common.Status.RUNNING
        dispatcher._theirs.update = lambda: called.append("theirs") or py_trees.common.Status.RUNNING

        self._tick_dispatcher(dispatcher, game)
        assert called == ["ours"]

    def test_penalty_yellow_calls_theirs_when_blue(self):
        from utama_core.strategy.referee.tree import _PenaltyDispatch

        data = _make_referee_data(command=RefereeCommand.PREPARE_PENALTY_YELLOW)
        # my_team_is_yellow=False, is_yellow_command=True → theirs
        game = _make_game(referee=data, my_team_is_yellow=False)
        dispatcher = _PenaltyDispatch(is_yellow_command=True, name="test")

        called = []
        dispatcher._ours.update = lambda: called.append("ours") or py_trees.common.Status.RUNNING
        dispatcher._theirs.update = lambda: called.append("theirs") or py_trees.common.Status.RUNNING

        self._tick_dispatcher(dispatcher, game)
        assert called == ["theirs"]

    def test_direct_free_blue_calls_ours_when_blue(self):
        from utama_core.strategy.referee.tree import _DirectFreeDispatch

        data = _make_referee_data(command=RefereeCommand.DIRECT_FREE_BLUE)
        # my_team_is_yellow=False, is_yellow_command=False → ours
        game = _make_game(referee=data, my_team_is_yellow=False)
        dispatcher = _DirectFreeDispatch(is_yellow_command=False, name="test")

        called = []
        dispatcher._ours.update = lambda: called.append("ours") or py_trees.common.Status.RUNNING
        dispatcher._theirs.update = lambda: called.append("theirs") or py_trees.common.Status.RUNNING

        self._tick_dispatcher(dispatcher, game)
        assert called == ["ours"]
