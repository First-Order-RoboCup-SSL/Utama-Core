"""Unit tests for the referee integration layer.

Tests cover:
  - RefereeData new fields (game_events, match_type, status_message) and custom __eq__
  - RefereeRefiner.refine injects data into GameFrame; deduplication logic
  - Game.referee property proxies correctly from CurrentGameFrame
  - strategy/referee/actions.py Step classes (Halt/Stop/BallPlacement/Kickoff/
    Penalty/DirectFree) — the geometry/positioning logic `RefereeOverride`
    reuses directly for the kernel-tactic model.
"""

import math
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
    ball: Ball | None = None,
) -> GameFrame:
    if friendly_robots is None:
        friendly_robots = {0: _robot(0)}
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly_robots,
        enemy_robots={},
        ball=ball or _ball(),
        referee=referee,
    )


def _make_game(
    friendly_robots=None,
    referee=None,
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = True,
    field_bounds: FieldBounds = STANDARD_FIELD_DIMS.full_field_bounds,
    ball: Ball | None = None,
) -> Game:
    frame = _make_game_frame(friendly_robots, referee, my_team_is_yellow, my_team_is_right, ball)
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
        # Target sits behind the ball (opposite the carry direction toward
        # designated_position) by _APPROACH_OFFSET, not on the ball's own
        # position -- see BallPlacementOursStep._APPROACH_OFFSET.
        ball_pos = Vector2D(game.ball.p.x, game.ball.p.y)
        designated = Vector2D(*game.referee.designated_position)
        oren = ball_pos.angle_to(designated)
        offset = referee_actions.BallPlacementOursStep._APPROACH_OFFSET
        assert captured[0][1].x == pytest.approx(ball_pos.x - offset * math.cos(oren))
        assert captured[0][1].y == pytest.approx(ball_pos.y - offset * math.sin(oren))
        assert captured[0][2] is True
        assert cmd_map[1] is not None

    def test_robot_without_ball_approaches_an_out_of_bounds_ball_with_offset(self, monkeypatch):
        """Even when the ball itself rests out of bounds, the chase target
        still applies the same behind-the-ball _APPROACH_OFFSET as the
        in-bounds case, clamped via `_clamp_to_field_or_ball` (which never
        clamps a point farther from the ball than it already is -- see that
        helper's own docstring). Genuinely out-of-bounds placement targets
        are not actually chased this way in a live sim run: strategy_runner.
        py teleports the ball straight onto designated_position (and holds
        it there until truly at rest -- see _TELEPORT_SETTLE_SPEED_MPS) the
        moment BALL_PLACEMENT_* begins, precisely because a robot can't
        physically retrieve an out-of-bounds ball in simulation. This test
        only exercises BallPlacementOursStep's own fallback geometry for
        whatever tick sees the ball still out of bounds (e.g. the one tick
        before that teleport lands).
        """
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords, dribbling))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {0: _robot(0, 4.0, 0.0)}
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        referee.designated_position = (0.9, -0.2)
        # Ball resting past the sideline on x only (STANDARD_FIELD_DIMS'
        # full_field_half_length is 4.5) -- a single-axis overshoot, the
        # shape actually seen live.
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=_ball(4.78, -1.04),
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
        ball_pos = Vector2D(game.ball.p.x, game.ball.p.y)
        designated = Vector2D(*game.referee.designated_position)
        oren = ball_pos.angle_to(designated)
        offset = referee_actions.BallPlacementOursStep._APPROACH_OFFSET
        approach = Vector2D(
            ball_pos.x - offset * math.cos(oren),
            ball_pos.y - offset * math.sin(oren),
        )
        expected = referee_actions._clamp_to_field_or_ball(approach, game, ball_pos)
        assert captured[0][1].x == pytest.approx(expected.x)
        assert captured[0][1].y == pytest.approx(expected.y)
        assert captured[0][2] is True

    def test_placer_choice_is_sticky_across_near_tied_distances(self, monkeypatch):
        """Regression for the same bug shape as `DirectFreeOursStep`'s kicker
        thrashing (roadmap item 15): a bare `min(..., key=distance)`
        recomputed fresh every tick flips identity on ordinary sim noise
        whenever two-plus robots are near-tied, resetting whoever newly
        "wins" to a standing start. `BallPlacementOursStep` must hold its
        placer choice via `Sticky` across ticks (the instance persists
        across ticks -- constructed once in `RefereeOverride.__init__`).
        """
        from utama_core.custom_referee import actions as referee_actions

        captured_placer_ids = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured_placer_ids.append(robot_id)
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        node = referee_actions.BallPlacementOursStep()
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        referee.designated_position = (5.0, 0.0)

        # Robots 1/3/5 start within 2cm of each other's distance to the ball
        # at (0, 0) -- near-tied, well under the reassign margin -- with tiny
        # per-tick jitter (sub-mm) simulating rsim sensor noise, exactly the
        # shape that made the naive `min()` flip every tick. None has the
        # ball, so every tick's placer takes the move()-to-ball branch.
        base_positions = {1: (2.00, 0.0), 3: (2.01, 0.0), 5: (2.02, 0.0)}
        jitter = [0.0, 0.001, -0.001, 0.0015, -0.0005]

        for dx in jitter:
            robots = {rid: _robot(rid, x + dx, 0.0) for rid, (x, y) in base_positions.items()}
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
            node.blackboard = _make_blackboard(game, cmd_map)
            node.update()

        # Every tick's placer must be the same robot once chosen -- no flip
        # from noise alone (all candidates stay within the reassign margin).
        assert len(set(captured_placer_ids)) == 1, (
            f"placer flipped across near-tied ticks: {captured_placer_ids} " "(expected the same robot id every tick)"
        )

    def test_placer_reassigns_when_a_robot_is_clearly_closer(self, monkeypatch):
        """The sticky placer choice must still yield to a genuinely closer
        robot -- hysteresis should suppress noise-level flips, not freeze the
        assignment forever."""
        from utama_core.custom_referee import actions as referee_actions

        captured_placer_ids = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured_placer_ids.append(robot_id)
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        node = referee_actions.BallPlacementOursStep()
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        referee.designated_position = (5.0, 0.0)

        def _run(robots):
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
            node.blackboard = _make_blackboard(game, cmd_map)
            node.update()

        # Tick 1: robot 1 is closest.
        _run({1: _robot(1, 2.0, 0.0), 3: _robot(3, 3.0, 0.0)})
        assert captured_placer_ids[-1] == 1

        # Tick 2: robot 3 is now far closer (clears the reassign margin).
        _run({1: _robot(1, 2.0, 0.0), 3: _robot(3, 0.5, 0.0)})
        assert captured_placer_ids[-1] == 3

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

    def test_release_withheld_while_ball_still_moving_fast_near_target(self, monkeypatch):
        """Regression: being within BALL_PLACEMENT_DONE_DISTANCE isn't enough
        to start the release countdown if the ball is still coasting in fast
        -- see BallPlacementOursStep._SETTLED_SPEED_MPS's docstring. A robot
        carrying the ball at ~1.4 m/s can't brake to a stop within the 0.15m
        done radius, so releasing there let the ball slide well past
        designated_position with nothing left to re-collect it, freezing
        BALL_PLACEMENT_YELLOW for the rest of a live match (confirmed
        2026-09-12). While the ball is within range but still fast, the step
        must keep driving the placer (move(), not empty_command()) rather
        than starting the release countdown.
        """
        from utama_core.custom_referee import actions as referee_actions

        move_captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            move_captured.append((robot_id, target_coords, dribbling))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        target = (1.5, 0.0)
        robots = {
            0: Robot(
                id=0,
                is_friendly=True,
                has_ball=True,
                p=Vector2D(1.4, 0.0),
                v=Vector2D(1.4, 0.0),
                a=Vector2D(0.0, 0.0),
                orientation=0.0,
            )
        }
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        referee.designated_position = target
        # Within BALL_PLACEMENT_DONE_DISTANCE (0.1m to go) but still moving
        # fast (1.4 m/s) -- the old distance-only gate would start releasing
        # here; the fix must not.
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=Ball(p=Vector3D(1.4, 0.0, 0.0), v=Vector3D(1.4, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
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

        assert node._release_started_at is None
        assert len(move_captured) >= 1
        assert move_captured[0][0] == 0

    def test_release_starts_once_ball_is_close_and_settled(self, monkeypatch):
        """Companion to the fast-ball case above: once the ball is both
        within BALL_PLACEMENT_DONE_DISTANCE and slow (<= _SETTLED_SPEED_MPS),
        the release countdown must actually start."""
        from utama_core.custom_referee import actions as referee_actions

        monkeypatch.setattr(referee_actions, "move", lambda *a, **k: ("move", a[2]))

        target = (1.5, 0.0)
        robots = {
            0: Robot(
                id=0,
                is_friendly=True,
                has_ball=True,
                p=Vector2D(1.4, 0.0),
                v=Vector2D(0.0, 0.0),
                a=Vector2D(0.0, 0.0),
                orientation=0.0,
            )
        }
        referee = _make_referee_data(command=RefereeCommand.BALL_PLACEMENT_YELLOW)
        referee.designated_position = target
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots={},
            ball=Ball(p=Vector3D(1.4, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
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

        assert node._release_started_at == 0.0

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
        # Placer approaches from behind the ball (opposite the carry
        # direction toward designated_position), not straight onto the
        # ball's own position -- see BallPlacementOursStep._APPROACH_OFFSET.
        assert placer_move[1] == Vector2D(
            game.ball.p.x - referee_actions.BallPlacementOursStep._APPROACH_OFFSET, game.ball.p.y
        )
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
        game = _make_game(
            friendly_robots=robots,
            referee=referee,
            my_team_is_yellow=True,
            my_team_is_right=True,
            ball=_ball(2.25, 0.0),
        )
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PreparePenaltyTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()
        keeper_target = next(target for robot_id, target in captured if robot_id == 1)
        support_targets = [target for robot_id, target in captured if robot_id != 1]

        assert keeper_target.x == pytest.approx(4.5)
        assert all(target.x > 0.0 for target in support_targets)

    def test_prepare_penalty_theirs_support_robots_stand_outside_the_keep_out_circle(self, monkeypatch):
        # decoy_and_overload_vs_low_block (2026-09-28): the line 0.4 m behind the
        # mark put a defender 0.44 m from the ball, KeepOutRule (0.5 m) fired
        # 1 s into every PREPARE_PENALTY, and the penalty was never taken.
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        robots = {rid: _robot(rid, 0.5 * rid, 0.0) for rid in range(6)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_PENALTY_BLUE)
        referee.yellow_team.goalkeeper = 0
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)
        node = referee_actions.PreparePenaltyTheirsStep()
        node.blackboard = _make_blackboard(game, _make_cmd_map(game))

        node.update()
        mark = Vector2D(2.25, 0.0)
        support_targets = [target for robot_id, target in captured if robot_id != 0]
        assert len(support_targets) == 5
        assert min((target - mark).mag() for target in support_targets) >= 1.0

    def test_prepare_penalty_theirs_goal_side_defender_walks_round_the_ball(self, monkeypatch):
        # decoy_and_overload_vs_low_block (2026-09-28): a defender starting between
        # the mark and its own goal drove straight through the ball to the line
        # behind it, 0.30 m from the ball for 0.5 s -- keep-out voided the penalty.
        from utama_core.custom_referee import actions as referee_actions

        captured = {}

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured[robot_id] = target_coords
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        mark = Vector2D(2.25, 0.0)
        robots = {0: _robot(0, 4.4, 0.0), 1: _robot(1, 3.1, 0.05)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_PENALTY_BLUE)
        referee.yellow_team.goalkeeper = 0
        game = _make_game(
            friendly_robots=robots,
            referee=referee,
            my_team_is_yellow=True,
            my_team_is_right=True,
            ball=_ball(mark.x, mark.y),
        )
        node = referee_actions.PreparePenaltyTheirsStep()
        node.blackboard = _make_blackboard(game, _make_cmd_map(game))

        node.update()
        start, target = Vector2D(3.1, 0.05), captured[1]
        seg = target - start
        t = min(1.0, max(0.0, (mark - start).dot(seg) / seg.dot(seg)))
        assert (start + seg * t - mark).mag() >= 0.5

    def test_prepare_penalty_ours_goal_side_teammate_walks_round_the_ball(self, monkeypatch):
        # Mirror of the defender case: a non-kicker of ours left goal-side of the
        # mark drove straight through the placed ball to its line 1 m behind it,
        # knocking the ball off the mark before the kick.
        from utama_core.custom_referee import actions as referee_actions

        captured = {}

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured[robot_id] = target_coords
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        mark = Vector2D(-2.25, 0.0)
        robots = {0: _robot(0, 4.4, 0.0), 1: _robot(1, -1.0, 0.0), 2: _robot(2, -3.1, 0.05)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_PENALTY_YELLOW)
        referee.yellow_team.goalkeeper = 0
        game = _make_game(
            friendly_robots=robots,
            referee=referee,
            my_team_is_yellow=True,
            my_team_is_right=True,
            ball=_ball(mark.x, mark.y),
        )
        node = referee_actions.PreparePenaltyOursStep()
        node.blackboard = _make_blackboard(game, _make_cmd_map(game))

        node.update()
        assert captured[1] == mark  # the kicker still goes straight to the mark
        start, target = Vector2D(-3.1, 0.05), captured[2]
        seg = target - start
        t = min(1.0, max(0.0, (mark - start).dot(seg) / seg.dot(seg)))
        assert (start + seg * t - mark).mag() >= 0.5


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
        # yellow_team.goalkeeper=2 (see _make_referee_data's default) — robot 2
        # is the real keeper here, not robot 0. It must be fully exempt from
        # formation (absent from cmd_map), not just excluded from kicker choice.
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
        assert cmd_map[2] is None, "goalkeeper must be exempt from kickoff formation entirely"

        # Kicker must be the lowest-ID non-keeper robot.
        kicker_target = next(target for robot_id, target in captured if robot_id == 0)
        support_target = next(target for robot_id, target in captured if robot_id == 1)

        assert kicker_target.distance_to(Vector2D(0.12, 0.0)) < 1e-9
        # Non-encroaching support robot heads to the field-scaled formation slot.
        assert support_target.x == pytest.approx(6.0 * (0.8 / 4.5))
        assert support_target.y == pytest.approx(4.0 * (0.5 / 3.0))
        assert support_target.distance_to(Vector2D(0.0, 0.0)) >= 0.5

    def test_prepare_kickoff_ours_uses_own_half_when_defending_left(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        # Support robot starts in its own half: from the other half its first target is
        # a detour waypoint round the centre circle (see TestDetourAroundCircle).
        robots = {
            0: _robot(0, 0.0, 0.0),
            1: _robot(1, -1.0, 1.0),
        }
        custom_bounds = FieldBounds(top_left=(-6.0, 4.0), bottom_right=(6.0, -4.0))
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_YELLOW)
        # yellow_team.goalkeeper=2 (see _make_referee_data's default), which
        # isn't even on the field here — both robots 0 and 1 are legitimate
        # outfield robots, so robot 0 (lowest ID) is the kicker.
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
        kicker_target = next(target for robot_id, target in captured if robot_id == 0)
        support_target = next(target for robot_id, target in captured if robot_id == 1)

        assert kicker_target.distance_to(Vector2D(-0.12, 0.0)) < 1e-9
        assert support_target.distance_to(Vector2D(0.0, 0.0)) >= 0.5
        assert support_target.x < 0.0, "support robot must stay on own half (defending left)"

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
        # my_team_is_right=True → own half is positive-x side.
        # yellow_team.goalkeeper=2 (see _make_referee_data's default): robot 2
        # is the real keeper and must be exempt from formation entirely.
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=True)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert 2 not in captured, "goalkeeper must be exempt from kickoff formation entirely"
        assert len(captured) == 2
        for _, target in captured.items():
            assert target.x > 0.0, f"Expected positive-x (own half right), got {target}"

    def test_all_robots_placed_on_own_half_left(self, monkeypatch):
        from utama_core.custom_referee import actions as referee_actions

        captured = {}

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured[robot_id] = target_coords
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        # Robots start in their own half, clear of the ball: from the other half the first
        # target is a detour waypoint round the centre circle (see TestDetourAroundCircle).
        robots = {i: _robot(i, -1.0 - i, 1.0) for i in range(3)}
        referee = _make_referee_data(command=RefereeCommand.PREPARE_KICKOFF_YELLOW)
        # my_team_is_right=False → own half is negative-x side.
        # yellow_team.goalkeeper=2 (see _make_referee_data's default): robot 2
        # is the real keeper and must be exempt from formation entirely.
        game = _make_game(friendly_robots=robots, referee=referee, my_team_is_yellow=True, my_team_is_right=False)
        cmd_map = _make_cmd_map(game)
        node = referee_actions.PrepareKickoffTheirsStep()
        node.blackboard = _make_blackboard(game, cmd_map)

        node.update()

        assert 2 not in captured, "goalkeeper must be exempt from kickoff formation entirely"
        assert len(captured) == 2
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

    def test_approach_point_stays_reachable_when_ball_is_near_opponent_box(self, monkeypatch):
        """Live-traced deadlock, 2026-09-05 (full-length tournament run
        `counter_press_vs_switch_of_play_Rk`): the kicker's approach point
        (computed from the ball toward the kick-target enemy, offset by
        `_APPROACH_OFFSET`) landed inside the opponent's defense area
        whenever the ball itself sat close to the box edge (0.035m away in
        the traced case). Clamping that point out of the box using the same
        `OPPONENT_DEFENSE_AREA_KEEP_DISTANCE` (0.25m) general restart
        positioning uses made the clamped target unreachable within
        `_KICK_READY_DISTANCE` (0.16m) -- the kicker converged to the
        clamped point and held there for the rest of the match, since
        `_KICK_READY_DISTANCE` never triggered `empty_command`. The fix uses
        a much smaller robot-radius clearance (matching `DefenseAreaRule`'s
        actual strict-boundary foul condition, not a keep-distance) and
        never pushes the target farther from the ball than it already was.

        Reproduces the traced geometry directly: `my_team_is_right=True`
        (opponent box on the left, x in [-4.5, -3.5]), ball at
        (-3.465, -0.888) -- 0.035m from the box's front edge -- with an
        enemy id=1 southeast of the ball so `_kick_target_enemy` picks the
        same kick direction traced live.
        """
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        ball_pos = Vector2D(-3.46505101, -0.88841382)
        robots = {3: _robot(3, -3.50031788, -1.46884941)}
        enemy_robots = {1: _robot(1, -2.76625651, -0.50512429)}
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots=robots,
            enemy_robots=enemy_robots,
            ball=Ball(Vector3D(ball_pos.x, ball_pos.y, 0.0), Vector3D(0, 0, 0), Vector3D(0, 0, 0)),
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
        target = captured[0][1]

        # The target must be legally outside the opponent's defense area
        # (front edge at x=-3.5 for this field/side)...
        assert target.x > -3.5
        # ...AND still within kicking range of the ball -- the deadlock this
        # test pins was a target that satisfied the first assertion but not
        # this one, so the kicker held at `target` forever.
        assert target.distance_to(ball_pos) <= referee_actions.DirectFreeOursStep._KICK_READY_DISTANCE

    def test_kicker_goes_round_the_ball_to_an_approach_point_on_its_far_side(self, monkeypatch):
        """The approach point is behind the ball from the kick direction; a kicker coming
        from the other side drove straight through the ball to reach it and pushed a free
        kick placed 0.25 m inside the goal line back onto the line, where the first touch
        put it out: 90 of 478 free kicks, tournament_20260927_223257."""
        from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
        from utama_core.custom_referee import actions as referee_actions

        captured = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured.append((robot_id, target_coords))
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        ball_pos = Vector2D(4.25, 2.43)
        kicker_pos = Vector2D(3.7, 2.43)
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={3: _robot(3, kicker_pos.x, kicker_pos.y)},
            enemy_robots={1: _robot(1, 0.0, 2.43)},  # kick straight back up the field, -x
            ball=_ball(ball_pos.x, ball_pos.y),
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
        node = referee_actions.DirectFreeOursStep()
        node.blackboard = _make_blackboard(game, _make_cmd_map(game))

        node.update()

        target = captured[0][1]
        seg = target - kicker_pos
        t = max(0.0, min(1.0, (ball_pos - kicker_pos).dot(seg) / seg.dot(seg)))
        closest = (kicker_pos + seg * t).distance_to(ball_pos)
        assert closest >= ROBOT_RADIUS + BALL_RADIUS

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

    def test_kicker_choice_is_sticky_across_near_tied_distances(self, monkeypatch):
        """Regression for the DIRECT_FREE congestion stall root-caused via
        `clear_danger_vs_clear_press_plus` (roadmap item 15): three robots
        within a couple cm of each other's distance to the ball flipped
        which one `min(..., key=distance)` called "closest" on ordinary
        rsim position noise, ~9 times/second, so no robot ever held the
        kicker role long enough to make progress. `DirectFreeOursStep` must
        reuse the same instance across ticks (it does — constructed once in
        `RefereeOverride.__init__`) and hold its choice via `Sticky`.
        """
        from utama_core.custom_referee import actions as referee_actions

        captured_kicker_ids = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured_kicker_ids.append(robot_id)
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        node = referee_actions.DirectFreeOursStep()
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_YELLOW)

        # Robots 1/3/5 start within 2cm of each other's distance to the ball
        # at (0, 0) — near-tied, well under the reassign margin — with tiny
        # per-tick jitter (sub-mm) simulating rsim sensor noise, exactly the
        # shape that made the naive `min()` flip every tick.
        base_positions = {1: (2.00, 0.0), 3: (2.01, 0.0), 5: (2.02, 0.0)}
        jitter = [0.0, 0.001, -0.001, 0.0015, -0.0005]

        for tick, dx in enumerate(jitter):
            robots = {rid: _robot(rid, x + dx, 0.0) for rid, (x, y) in base_positions.items()}
            game = _make_game(friendly_robots=robots, referee=referee)
            cmd_map = _make_cmd_map(game)
            node.blackboard = _make_blackboard(game, cmd_map)
            node.update()

        # Every tick's kicker must be the same robot once chosen — no flip
        # from noise alone (all candidates stay within the reassign margin).
        assert len(set(captured_kicker_ids)) == 1, (
            f"kicker flipped across near-tied ticks: {captured_kicker_ids} " "(expected the same robot id every tick)"
        )

    def test_kicker_reassigns_when_a_robot_is_clearly_closer(self, monkeypatch):
        """The sticky kicker choice must still yield to a genuinely closer
        robot — hysteresis should suppress noise-level flips, not freeze the
        assignment forever."""
        from utama_core.custom_referee import actions as referee_actions

        captured_kicker_ids = []

        def fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
            captured_kicker_ids.append(robot_id)
            return ("move", robot_id)

        monkeypatch.setattr(referee_actions, "move", fake_move)

        node = referee_actions.DirectFreeOursStep()
        referee = _make_referee_data(command=RefereeCommand.DIRECT_FREE_YELLOW)

        # Tick 1: robot 1 is closest.
        robots = {1: _robot(1, 2.0, 0.0), 3: _robot(3, 3.0, 0.0)}
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node.blackboard = _make_blackboard(game, cmd_map)
        node.update()
        assert captured_kicker_ids[-1] == 1

        # Tick 2: robot 3 is now far closer (clears the reassign margin) but
        # still well outside _APPROACH_READY_DISTANCE, so it takes the move()
        # branch rather than turn_on_spot()/empty_command().
        robots = {1: _robot(1, 2.0, 0.0), 3: _robot(3, 0.5, 0.0)}
        game = _make_game(friendly_robots=robots, referee=referee)
        cmd_map = _make_cmd_map(game)
        node.blackboard = _make_blackboard(game, cmd_map)
        node.update()
        assert captured_kicker_ids[-1] == 3


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


class TestDetourAroundCircle:
    """`_detour_around_circle`: robots crossing the ball's keep-out circle go round it."""

    def _detour(self, start, target):
        from utama_core.custom_referee.actions import _detour_around_circle

        return _detour_around_circle(Vector2D(*start), Vector2D(*target), Vector2D(0.0, 0.0), 0.8)

    def test_clear_line_keeps_the_target(self):
        assert self._detour((-1.0, 1.0), (1.0, 1.0)) == Vector2D(1.0, 1.0)

    def test_line_through_the_circle_goes_round_on_the_targets_side(self):
        w = self._detour((-0.9, -0.05), (0.8, 0.4))
        assert w.mag() > 0.8  # outside the circle
        assert w.y > 0.0  # round the side the target is on
        assert w.x > -0.9  # and making progress toward it

    def test_leg_to_the_waypoint_stays_outside_from_the_edge(self):
        """From a robot parked on the edge, the straight leg to the waypoint must not cut
        back inside, or the push-out and the detour fight and the robot parks there."""
        start = Vector2D(-0.8, 0.0)
        w = self._detour((start.x, start.y), (0.8, 0.0))
        for k in range(11):
            p = start + (w - start) * (k / 10)
            assert p.mag() >= 0.8 - 1e-9
