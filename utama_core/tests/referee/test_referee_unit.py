"""Unit tests for the referee integration layer.

Tests cover:
  - RefereeData new fields (game_events, match_type, status_message) and custom __eq__
  - RefereeRefiner.refine injects data into GameFrame; deduplication logic
  - Game.referee property proxies correctly from CurrentGameFrame
  - strategy/referee/actions.py Step classes (Halt/Stop/BallPlacement/Kickoff/
    Penalty/DirectFree) — the geometry/positioning logic `kernel.RefereeOverride`
    reuses directly for the kernel-tactic model.
"""

from types import SimpleNamespace

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.data_processing.refiners.referee import RefereeRefiner
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.entities.game.game import Game
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import GameHistory
from utama_core.entities.game.robot import Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _team_info(goalkeeper: int = 0, can_place_ball: bool = True) -> TeamInfo:
    return TeamInfo(
        name="TestTeam",
        score=0,
        red_cards=0,
        yellow_card_times=[],
        yellow_cards=0,
        timeouts=0,
        timeout_time=0,
        goalkeeper=goalkeeper,
        can_place_ball=can_place_ball,
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
    field_bounds: FieldBounds = STANDARD_FIELD_DIMS.full_field_bounds,
) -> Game:
    frame = _make_game_frame(friendly_robots, referee, my_team_is_yellow, my_team_is_right)
    history = GameHistory(10)
    return Game(
        past=history,
        current=frame,
        field=Field(my_team_is_right=my_team_is_right, field_dims=STANDARD_FIELD_DIMS, field_bounds=field_bounds),
    )


def _make_blackboard(game: Game, cmd_map=None):
    """Construct a minimal SimpleNamespace blackboard as used by the referee Step classes."""
    bb = SimpleNamespace()
    bb.game = game
    bb.cmd_map = cmd_map if cmd_map is not None else {}
    bb.motion_controller = SimpleNamespace(calculate=lambda **kwargs: (Vector2D(0.0, 0.0), 0.0))
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
# HaltStep and StopStep — basic output verification
# ---------------------------------------------------------------------------


def _make_cmd_map(game: Game) -> dict:
    return {rid: None for rid in game.friendly_robots}


class TestHaltAndStopStep:
    def _run_step(self, step_class, game: Game) -> dict:
        cmd_map = _make_cmd_map(game)
        bb = _make_blackboard(game, cmd_map)
        node = step_class()
        node.blackboard = bb
        node.update()
        return cmd_map

    def test_halt_writes_to_all_robots(self):
        from utama_core.custom_referee.actions import HaltStep

        robots = {0: _robot(0), 1: _robot(1)}
        game = _make_game(friendly_robots=robots, referee=_make_referee_data())
        cmd_map = self._run_step(HaltStep, game)
        assert set(cmd_map.keys()) == {0, 1}
        for rid in robots:
            assert cmd_map[rid] is not None

    def test_stop_writes_to_all_robots(self):
        from utama_core.custom_referee.actions import StopStep

        robots = {0: _robot(0), 1: _robot(1), 2: _robot(2)}
        game = _make_game(friendly_robots=robots, referee=_make_referee_data())
        cmd_map = self._run_step(StopStep, game)
        assert set(cmd_map.keys()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# BallPlacementOursStep — fetch ball before target placement
# ---------------------------------------------------------------------------


class TestBallPlacementOursStep:
    def test_robot_without_ball_moves_to_ball_first(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords, dribbling))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 2.0, 2.0),
        }
        referee = _make_referee_data(
            command=RefereeCommand.BALL_PLACEMENT_YELLOW,
        )
        referee.designated_position = (1.5, 1.5)
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=_ball(0.2, 0.0),
            referee=referee,
        )
        game = Game(
            past=GameHistory(10),
            current=frame,
            field=Field(
                my_team_is_right=True,
                field_dims=STANDARD_FIELD_DIMS,
                field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
            ),
        )

        cmd_map = _make_cmd_map(game)
        node = referee_actions.BallPlacementOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert captured[0][0] == 0
        assert captured[0][1].x == game.ball.p.x
        assert captured[0][1].y == game.ball.p.y
        assert captured[0][2] is True
        assert cmd_map[1] is not None

    def test_robot_with_ball_moves_to_designated_position(self, monkeypatch):
        import math

        from utama_core.custom_referee import actions as referee_actions

        move_captured = []
        turn_captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            move_captured.append((robot_id, target_coords, dribbling))
            return ("move", robot_id)

        def fake_turn_on_spot(game, motion_controller, robot_id, target_oren, dribbling=False):
            turn_captured.append((robot_id, dribbling))
            return ("turn", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)
        monkeypatch.setattr(referee_actions, "turn_on_spot", fake_turn_on_spot)

        target = (1.5, -0.5)
        # Orient the robot to face the target so face_error < threshold → move branch.
        facing = math.atan2(target[1], target[0])
        robots = {
            0: Robot(
                id=0,
                is_friendly=True,
                has_ball=True,
                p=Vector2D(0.0, 0.0),
                v=Vector2D(0.0, 0.0),
                a=Vector2D(0.0, 0.0),
                orientation=facing,
            )
        }
        referee = _make_referee_data(
            command=RefereeCommand.BALL_PLACEMENT_YELLOW,
        )
        referee.designated_position = target
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=_ball(0.0, 0.0),
            referee=referee,
        )
        game = Game(
            past=GameHistory(10),
            current=frame,
            field=Field(
                my_team_is_right=True,
                field_dims=STANDARD_FIELD_DIMS,
                field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
            ),
        )

        cmd_map = _make_cmd_map(game)
        node = referee_actions.BallPlacementOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        # Robot already faces target → should call move (not turn_on_spot).
        assert len(move_captured) >= 1
        assert move_captured[0][0] == 0
        assert move_captured[0][1] == Vector2D(1.5, -0.5)
        assert move_captured[0][2] is True

    def test_non_placing_teammate_clears_from_ball(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords, dribbling))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 0.1, 0.0),
        }
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        referee.designated_position = (1.5, 0.0)
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)

        cmd_map = _make_cmd_map(game)
        node = referee_actions.BallPlacementOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 2
        placer_move = next(item for item in captured if item[0] == 0)
        support_move = next(item for item in captured if item[0] == 1)
        assert placer_move[1] == Vector2D(game.ball.p.x, game.ball.p.y)
        assert support_move[1] == Vector2D(0.8, 0.0)
        assert support_move[2] is False


# ---------------------------------------------------------------------------
# Keep-out retreat and penalty positioning
# ---------------------------------------------------------------------------


class TestRefereeKeepOutRetreat:
    def test_stop_moves_only_robots_inside_keep_out_radius(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords, dribbling))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 1.0, 0.0),
        }
        game = _make_game(friendly_robots=robots, referee=_make_referee_data(command=RefereeCommand.STOP))
        cmd_map = _make_cmd_map(game)
        node = referee_actions.StopStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 1
        assert captured[0][0] == 0
        assert captured[0][1] == Vector2D(0.8, 0.0)
        assert cmd_map[1] is not None

    def test_stop_clears_robot_from_opponent_defense_area(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, -4.3, 0.0),
            1: _robot(1, 1.0, 0.0),
        }
        referee = _make_referee_data(command=RefereeCommand.STOP)
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=_ball(2.0, 0.0),
            referee=referee,
        )
        game = Game(
            past=GameHistory(10),
            current=frame,
            field=Field(
                my_team_is_right=True,
                field_dims=STANDARD_FIELD_DIMS,
                field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
            ),
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.StopStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 1
        assert captured[0][0] == 0
        assert captured[0][1] == Vector2D(-3.25, 0.0)

    def test_ball_placement_theirs_clears_encroaching_robot(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.1, 0.0),
            1: _robot(1, 1.0, 0.0),
        }
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.BallPlacementTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 1
        assert captured[0][0] == 0
        assert captured[0][1] == Vector2D(0.8, 0.0)

    def test_ball_placement_theirs_clears_robot_from_designated_position(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 1.0, 1.0),
            1: _robot(1, -1.0, 0.0),
        }
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_BLUE)
        referee.designated_position = (1.0, 1.0)
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=_ball(0.0, 0.0),
            referee=referee,
        )
        game = Game(
            past=GameHistory(10),
            current=frame,
            field=Field(
                my_team_is_right=True,
                field_dims=STANDARD_FIELD_DIMS,
                field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
            ),
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.BallPlacementTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 1
        assert captured[0][0] == 0
        assert captured[0][1] == Vector2D(1.8, 1.0)

    def test_direct_free_theirs_clears_encroaching_robot(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.2, 0.0),
            1: _robot(1, -1.0, 0.0),
        }
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 1
        assert captured[0][0] == 0
        assert captured[0][1] == Vector2D(0.8, 0.0)


class TestPenaltyPositioning:
    def test_prepare_penalty_ours_kicker_stays_on_attacking_half(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 1.0, 0.0),
        }
        referee = _make_referee_data(command=RefereeCommand.PREPARE_PENALTY_YELLOW)
        referee.yellow_team.goalkeeper = 1
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PreparePenaltyOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()
        kicker_target = next(target for robot_id, target in captured if robot_id == 0)

        assert kicker_target.x == pytest.approx(-2.25)
        assert kicker_target.x < 0.0

    def test_prepare_penalty_theirs_support_robots_stay_on_our_half(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 0.5, 0.0),
            2: _robot(2, -0.5, 0.0),
        }
        referee = _make_referee_data(command=RefereeCommand.PREPARE_PENALTY_BLUE)
        referee.yellow_team.goalkeeper = 1
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PreparePenaltyTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()
        keeper_target = next(target for robot_id, target in captured if robot_id == 1)
        support_targets = [target for robot_id, target in captured if robot_id != 1]

        assert keeper_target.x == pytest.approx(4.5)
        assert all(target.x > 0.0 for target in support_targets)


class TestVariableFieldScaling:
    def test_prepare_kickoff_ours_scales_support_positions_with_field_bounds(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 1.0, 0.0),
            2: _robot(2, 2.0, 0.0),
        }
        custom_bounds = FieldBounds(top_left=(-6.0, 4.0), bottom_right=(6.0, -4.0))
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_YELLOW)
        game = _make_game(
            friendly_robots=robots,
            referee=referee,
            my_team_is_yellow=True,
            my_team_is_right=True,
            field_bounds=custom_bounds,
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()
        # Kicker must be the lowest-ID *outfield* robot — robot 0 is the pinned
        # goalkeeper and must not leave its line to take kickoffs (it is a
        # support robot here, not the kicker).
        kicker_target = next(target for robot_id, target in captured if robot_id == 1)
        keeper_target = next(target for robot_id, target in captured if robot_id == 0)
        second_support_target = next(target for robot_id, target in captured if robot_id == 2)

        assert kicker_target.distance_to(Vector2D(0.12, 0.0)) < 1e-9
        # The keeper starts ON the ball (0, 0): it is encroaching, so it is
        # cleared straight out of the keep-out zone along the own-half
        # fallback direction instead of heading to its formation slot (which
        # would cut across the exclusion zone).
        assert keeper_target.distance_to(Vector2D(0.8, 0.0)) < 1e-9
        assert keeper_target.distance_to(Vector2D(0.0, 0.0)) >= 0.5
        # Non-encroaching support robots head to the field-scaled formation slots.
        assert second_support_target.x == pytest.approx(6.0 * (0.8 / 4.5))
        assert second_support_target.y == pytest.approx(-4.0 * (0.5 / 3.0))
        assert second_support_target.distance_to(Vector2D(0.0, 0.0)) >= 0.5

    def test_prepare_kickoff_ours_uses_own_half_when_defending_left(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 1.0, 0.0),
        }
        custom_bounds = FieldBounds(top_left=(-6.0, 4.0), bottom_right=(6.0, -4.0))
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_YELLOW)
        game = _make_game(
            friendly_robots=robots,
            referee=referee,
            my_team_is_yellow=True,
            my_team_is_right=False,
            field_bounds=custom_bounds,
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()
        # Robot 1 is the lowest-ID outfield robot → the kicker; the keeper
        # (robot 0) is a support robot.
        kicker_target = next(target for robot_id, target in captured if robot_id == 1)
        # Keeper starts ON the ball (0, 0): cleared straight out of the
        # keep-out zone toward own half (negative x when defending left).
        keeper_target = next(target for robot_id, target in captured if robot_id == 0)

        assert kicker_target.distance_to(Vector2D(-0.12, 0.0)) < 1e-9
        assert keeper_target.distance_to(Vector2D(-0.8, 0.0)) < 1e-9
        assert keeper_target.distance_to(Vector2D(0.0, 0.0)) >= 0.5

    def test_prepare_penalty_ours_scales_penalty_mark_with_field_bounds(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 1.0, 0.0),
        }
        custom_bounds = FieldBounds(top_left=(-6.0, 4.0), bottom_right=(6.0, -4.0))
        referee = _make_referee_data(command=RefereeCommand.PREPARE_PENALTY_YELLOW)
        referee.yellow_team.goalkeeper = 1
        game = _make_game(
            friendly_robots=robots,
            referee=referee,
            my_team_is_yellow=True,
            my_team_is_right=True,
            field_bounds=custom_bounds,
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PreparePenaltyOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()
        kicker_target = next(target for robot_id, target in captured if robot_id == 0)

        assert kicker_target.x == pytest.approx(-3.0)
        assert kicker_target.y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PrepareKickoffTheirsStep
# ---------------------------------------------------------------------------


class TestPrepareKickoffTheirsStep:
    def test_returns_running(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **kw: ("move",))

        robots = {0: _robot(0, 0.0, 0.0)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

    def test_all_robots_placed_on_own_half_right(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = {}

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured[robot_id] = target_coords
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {i: _robot(i, float(i), 0.0) for i in range(3)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_BLUE)
        # my_team_is_right=True → own half is positive-x side
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 3
        for _, target in captured.items():
            assert target.x > 0.0, f"Expected positive-x (own half right), got {target}"

    def test_all_robots_placed_on_own_half_left(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = {}

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured[robot_id] = target_coords
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {i: _robot(i, float(i), 0.0) for i in range(3)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_YELLOW)
        # my_team_is_right=False → own half is negative-x side
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=False)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 3
        for _, target in captured.items():
            assert target.x < 0.0, f"Expected negative-x (own half left), got {target}"

    def test_positions_outside_centre_circle(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {i: _robot(i, 0.0, 0.0) for i in range(6)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        for _, target in captured:
            dist = target.distance_to(Vector2D(0.0, 0.0))
            assert dist >= 0.5, f"Target {target} inside centre circle"

    def test_scales_with_custom_field_bounds(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {0: _robot(0, 0.0, 0.0), 1: _robot(1, 0.0, 0.0)}
        custom_bounds = FieldBounds(top_left=(-6.0, 4.0), bottom_right=(6.0, -4.0))
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_BLUE)
        game = _make_game(
            friendly_robots=robots,
            referee=referee,
            my_team_is_yellow=True,
            my_team_is_right=True,
            field_bounds=custom_bounds,
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        # Positions must be on own half and outside centre circle
        for _, target in captured:
            assert target.x > 0.0
            assert target.distance_to(Vector2D(0.0, 0.0)) >= 0.5


# ---------------------------------------------------------------------------
# DirectFreeOursStep
# ---------------------------------------------------------------------------


class TestDirectFreeOursStep:
    def test_returns_running(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **kw: ("move",))

        robots = {0: _robot(0, 0.0, 0.0)}
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_YELLOW)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

    def test_kicker_is_closest_robot_to_ball(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        # Robot 1 is closer to the ball at (1.0, 0.0)
        robots = {
            0: _robot(0, -2.0, 0.0),
            1: _robot(1, 1.2, 0.0),
        }
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=_ball(1.0, 0.0),
            referee=_make_referee_data(command=RefereeCommand.DIRECT_FREE_YELLOW),
        )
        game = Game(
            past=GameHistory(10),
            current=frame,
            field=Field(
                my_team_is_right=True,
                field_dims=STANDARD_FIELD_DIMS,
                field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
            ),
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        kicker_entries = [entry for entry in captured if entry[0] == 1]
        assert len(kicker_entries) == 1
        # Target is the approach point behind the ball, not the ball itself —
        # verify it is closer to the ball than robot 1's starting position.
        ball_pos = Vector2D(1.0, 0.0)
        robot1_start = Vector2D(1.2, 0.0)
        target = kicker_entries[0][1]
        assert target.distance_to(ball_pos) < robot1_start.distance_to(ball_pos)

    def test_kicker_moves_toward_ball(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {0: _robot(0, 0.5, 0.0)}
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=_ball(2.0, 1.0),
            referee=_make_referee_data(command=RefereeCommand.DIRECT_FREE_YELLOW),
        )
        game = Game(
            past=GameHistory(10),
            current=frame,
            field=Field(
                my_team_is_right=True,
                field_dims=STANDARD_FIELD_DIMS,
                field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
            ),
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 1
        assert captured[0][0] == 0
        # Target is the approach point behind the ball, not the ball itself —
        # verify it is closer to the ball than the robot's starting position.
        ball_pos = Vector2D(2.0, 1.0)
        robot_start = Vector2D(0.5, 0.0)
        target = captured[0][1]
        assert target.distance_to(ball_pos) < robot_start.distance_to(ball_pos)

    def test_non_kicker_robots_get_stop_command(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **kw: ("move",))

        # Robot 0 is closest to ball at (0.0, 0.0); robots 1 and 2 are farther
        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, 2.0, 0.0),
            2: _robot(2, -3.0, 1.0),
        }
        game = _make_game(
            friendly_robots=robots,
            referee=_make_referee_data(command=RefereeCommand.DIRECT_FREE_YELLOW),
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        # Robots 1 and 2 must have been given the empty (stop) command, not a move
        assert cmd_map[1] is not None
        assert cmd_map[2] is not None
        # The stop command is the empty_command tuple; move returns ("move", robot_id)
        assert cmd_map[1] != ("move", 1)
        assert cmd_map[2] != ("move", 2)

    def test_writes_command_for_every_robot(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **kw: ("move",))

        robots = {i: _robot(i, float(i), 0.0) for i in range(4)}
        game = _make_game(
            friendly_robots=robots,
            referee=_make_referee_data(command=RefereeCommand.DIRECT_FREE_YELLOW),
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeOursStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert set(cmd_map.keys()) == set(robots.keys())
        for v in cmd_map.values():
            assert v is not None


# ---------------------------------------------------------------------------
# DirectFreeTheirsStep — comprehensive keep-out coverage
# ---------------------------------------------------------------------------


class TestDirectFreeTheirsStep:
    def test_returns_running(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **kw: ("move",))

        robots = {0: _robot(0, 2.0, 0.0)}
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

    def test_robot_outside_keep_out_stays_put(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **kw: ("move",))

        robots = {0: _robot(0, 2.0, 0.0)}  # well outside 0.8 m from ball at (0,0)
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        # No move was needed; cmd_map entry should be the empty (stop) command
        assert cmd_map[0] is not None
        assert cmd_map[0] != ("move", 0)

    def test_multiple_robots_only_encroaching_ones_move(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        # Ball at (0, 0); robot 0 inside keep-out, robots 1 & 2 outside
        robots = {
            0: _robot(0, 0.1, 0.0),
            1: _robot(1, 1.0, 0.0),
            2: _robot(2, -1.5, 0.5),
        }
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        moved_ids = [rid for rid, _ in captured]
        assert moved_ids == [0]
        assert cmd_map[1] is not None
        assert cmd_map[2] is not None

    def test_encroaching_robot_projected_to_keep_out_boundary(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        # Ball at origin; robot dead on the x-axis at 0.3 m (inside 0.8)
        robots = {0: _robot(0, 0.3, 0.0)}
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert len(captured) == 1
        assert captured[0][1] == pytest.approx(Vector2D(0.8, 0.0))

    def test_all_robots_get_commands(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **kw: ("move",))

        robots = {i: _robot(i, float(i) * 0.5 - 1.0, 0.0) for i in range(5)}
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_BLUE)
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.DirectFreeTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert set(cmd_map.keys()) == set(robots.keys())
        for v in cmd_map.values():
            assert v is not None
