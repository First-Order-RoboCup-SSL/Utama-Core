"""Tests for the columnar replay format (`columnar_writer`/`columnar_reader`).

See `columnar_writer`'s module docstring for why this format exists: bulk
replay analysis only ever wants numeric arrays over a match's time axis,
and reconstructing one `GameFrame` per tick from pickle just to read a few
floats off it dominates load time on a full-length match.
"""

from __future__ import annotations

import pytest

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.columnar_reader import load_columnar_replay
from utama_core.replay.columnar_writer import (
    ColumnarReplayWriter,
    ColumnarReplayWriterConfig,
)


def _robot(rid: int, x: float, y: float, friendly: bool, has_ball: bool = False) -> Robot:
    return Robot(
        id=rid,
        is_friendly=friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0.1, 0.2),
        a=Vector2D(0.0, 0.0),
        orientation=1.5,
    )


def _referee(command: RefereeCommand, designated_position=None, **rare_fields) -> RefereeData:
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF_PRE,
        stage_time_left=0.0,
        blue_team=None,
        yellow_team=None,
        designated_position=designated_position,
        **rare_fields,
    )


def _make_writer(tmp_path, replay_name="test_replay", checkpoint_every_s=0):
    """A `ColumnarReplayWriter` writing under `tmp_path` instead of the
    real `REPLAY_BASE_PATH`, following the same isolation approach
    `test_stuck_detector.py` uses (write directly under `tmp_path`, don't
    exercise the real-directory path-resolution logic in a unit test)."""
    cfg = ColumnarReplayWriterConfig(
        replay_name=replay_name, overwrite_existing=True, checkpoint_every_s=checkpoint_every_s
    )
    return ColumnarReplayWriter(
        cfg, my_team_is_yellow=True, exp_friendly=6, exp_enemy=6, path=tmp_path / f"{replay_name}.npz"
    )


def _write(tmp_path, frames, replay_name="test_replay"):
    writer = _make_writer(tmp_path, replay_name)
    for f in frames:
        writer.write_frame(f)
    writer.close()
    return load_columnar_replay(writer.path)


def test_round_trip_preserves_robot_and_ball_state(tmp_path):
    frame = GameFrame(
        ts=0.1,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={3: _robot(3, 1.1, 1.1, True), 5: _robot(5, 2.0, 2.0, True, has_ball=True)},
        enemy_robots={2: _robot(2, -1.0, -1.0, False)},
        ball=Ball(p=Vector3D(1.0, 2.0, 0.1), v=Vector3D(0.5, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
        referee=None,
    )
    replay = _write(tmp_path, [frame])
    got = replay.frame_at(0)

    assert got.ts == pytest.approx(0.1)
    assert set(got.friendly_robots) == {3, 5}
    assert got.friendly_robots[3].p == frame.friendly_robots[3].p
    assert got.friendly_robots[5].has_ball is True
    assert got.friendly_robots[3].has_ball is False
    assert set(got.enemy_robots) == {2}
    assert got.ball.p == frame.ball.p
    assert got.ball.v == frame.ball.v
    assert got.referee is None


def test_round_trip_handles_changing_robot_ids_across_ticks(tmp_path):
    """Mirrors `test_replay.py`'s `test_non_sequential_robot_ids`: a robot
    id present at one tick can be absent at another, and a new id can
    appear later — the roster isn't fixed at the first tick."""
    frames = [
        GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={1: _robot(1, 1.0, 1.0, True)},
            enemy_robots={},
            ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        ),
        GameFrame(
            ts=0.1,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={3: _robot(3, 1.1, 1.1, True), 5: _robot(5, 2.0, 2.0, True)},
            enemy_robots={2: _robot(2, -1.0, -1.0, False)},
            ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        ),
    ]
    replay = _write(tmp_path, frames)

    got0 = replay.frame_at(0)
    got1 = replay.frame_at(1)
    assert set(got0.friendly_robots) == {1}
    assert set(got0.enemy_robots) == set()
    assert set(got1.friendly_robots) == {3, 5}
    assert set(got1.enemy_robots) == {2}
    # Robot 1 (absent at tick 1) must not leak into tick 1's roster, and
    # robots 3/5 (absent at tick 0) must not leak into tick 0's roster.
    assert 1 not in got1.friendly_robots
    assert 3 not in got0.friendly_robots


def test_round_trip_preserves_dense_referee_fields(tmp_path):
    frames = [
        GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={1: _robot(1, 0, 0, True)},
            enemy_robots={},
            ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
            referee=_referee(RefereeCommand.NORMAL_START, designated_position=(1.0, 2.0)),
        ),
        GameFrame(
            ts=1.0 / 60,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={1: _robot(1, 0, 0, True)},
            enemy_robots={},
            ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
            referee=_referee(RefereeCommand.STOP),
        ),
    ]
    replay = _write(tmp_path, frames)

    got0 = replay.frame_at(0)
    got1 = replay.frame_at(1)
    assert got0.referee.referee_command == RefereeCommand.NORMAL_START
    assert got0.referee.designated_position == (1.0, 2.0)
    assert got1.referee.referee_command == RefereeCommand.STOP
    assert got1.referee.designated_position is None


def test_round_trip_preserves_rare_referee_fields_via_sparse_sidecar(tmp_path):
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, 0, 0, True)},
        enemy_robots={},
        ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        referee=_referee(RefereeCommand.STOP, status_message="Blue attacker in yellow defense area"),
    )
    replay = _write(tmp_path, [frame])

    assert 0 in replay.sparse_referee
    got = replay.frame_at(0)
    assert got.referee.status_message == "Blue attacker in yellow defense area"


def test_frames_in_range_matches_frame_at_over_the_window(tmp_path):
    frames = [
        GameFrame(
            ts=i / 60,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={1: _robot(1, float(i), 0.0, True)},
            enemy_robots={},
            ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        )
        for i in range(10)
    ]
    replay = _write(tmp_path, frames)

    windowed = replay.frames_in_range(2 / 60, 5 / 60)
    assert [f.ts for f in windowed] == [f.ts for f in frames[2:6]]


def test_checkpoint_flush_leaves_a_valid_partial_replay(tmp_path):
    """A mid-match crash should leave the last checkpoint loadable, not an
    empty or corrupt file — this is the durability property that replaces
    the old format's per-frame flush."""
    writer = _make_writer(tmp_path, "checkpoint_test", checkpoint_every_s=0.01)

    for i in range(5):
        writer.write_frame(
            GameFrame(
                ts=i * 0.02,
                my_team_is_yellow=True,
                my_team_is_right=True,
                friendly_robots={1: _robot(1, float(i), 0.0, True)},
                enemy_robots={},
                ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
            )
        )
    # Simulate a crash: never call close(). A checkpoint should already
    # have fired (checkpoint_every_s=0.01, ticks 0.02s apart).
    assert writer.path.exists(), "expected at least one periodic checkpoint to have been written"
    replay = load_columnar_replay(writer.path)
    assert replay.n_ticks >= 1
