"""Tests for the columnar replay format (`columnar_writer`/`columnar_reader`).

See `columnar_writer`'s module docstring for why this format exists: bulk
replay analysis only ever wants numeric arrays over a match's time axis,
and reconstructing one `GameFrame` per tick from pickle just to read a few
floats off it dominates load time on a full-length match.
"""

from __future__ import annotations

import dataclasses

import pytest

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.game.team_info import TeamInfo
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
    assert (got.friendly_robots[3].p.x, got.friendly_robots[3].p.y) == pytest.approx((1.1, 1.1))  # stored float32
    assert got.friendly_robots[5].has_ball is True
    assert got.friendly_robots[3].has_ball is False
    assert set(got.enemy_robots) == {2}
    assert tuple(got.ball.p) == pytest.approx(tuple(frame.ball.p))
    assert tuple(got.ball.v) == pytest.approx(tuple(frame.ball.v))
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


def test_sparse_sidecar_stores_only_changes_and_rebuilds_every_tick(tmp_path):
    """A live referee message changes every tick only in its clocks: the sidecar keeps the
    ticks where anything else changes (or a clock jumps), and the reader advances the clocks."""
    yellow = TeamInfo("Yellow")  # mutated in place below, as a referee may do
    blue = TeamInfo("Blue")
    dt = 1.0 / 60

    def referee_at(i: int) -> RefereeData:
        ts = i * dt
        if i == 90:
            yellow.score = 1
        return RefereeData(
            source_identifier="custom_referee",
            time_sent=ts,
            time_received=ts,
            referee_command=RefereeCommand.STOP if i < 60 else RefereeCommand.NORMAL_START,
            referee_command_timestamp=0.0 if i < 60 else 1.0,
            stage=Stage.NORMAL_FIRST_HALF_PRE,
            stage_time_left=_stage_clock(i),
            blue_team=blue,
            yellow_team=yellow,
            status_message="Ball left the field" if 30 <= i < 40 else None,
        )

    def _stage_clock(i: int) -> float:
        # stopped (paused) until tick 60, then counting down; a jump at tick 100
        played = max(0, i - 60) * dt
        return 300.0 - played if i < 100 else 200.0 - (i - 100) * dt

    writer = _make_writer(tmp_path)
    for i in range(120):  # one frame at a time, as in a match: the in-place score change lands at tick 90
        writer.write_frame(
            GameFrame(
                ts=i * dt,
                my_team_is_yellow=True,
                my_team_is_right=True,
                friendly_robots={1: _robot(1, 0, 0, True)},
                enemy_robots={},
                ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
                referee=referee_at(i),
            )
        )
    writer.close()
    replay = load_columnar_replay(writer.path)

    assert sorted(replay.sparse_referee) == [0, 30, 40, 60, 90, 100]
    for i in range(120):
        got = replay.frame_at(i).referee
        ts = i * dt
        assert got.referee_command == (RefereeCommand.STOP if i < 60 else RefereeCommand.NORMAL_START)
        assert got.status_message == ("Ball left the field" if 30 <= i < 40 else None)
        assert got.yellow_team.score == (1 if i >= 90 else 0)
        assert got.time_sent == pytest.approx(ts)
        assert got.stage_time_left == pytest.approx(_stage_clock(i))


def test_state_arrays_are_stored_as_float32_and_read_back_as_float64(tmp_path):
    """Half the bytes: float32 keeps positions to well under a micrometre on a 12 m field, far
    below vision noise. Timestamps stay float64 (tiny, and frame lookups key on them)."""
    import numpy as np

    frame = GameFrame(
        ts=12.3456789,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, 1.23456789, -2.5, True)},
        enemy_robots={2: _robot(2, -3.0, 0.5, False)},
        ball=Ball(p=Vector3D(1.0, 2.0, 0.1), v=Vector3D(0.5, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
        referee=_referee(RefereeCommand.STOP, designated_position=(1.5, -0.25)),
    )
    writer = _make_writer(tmp_path)
    writer.write_frame(frame)
    writer.close()

    with np.load(writer.path) as stored:
        for key in ("friendly_p", "friendly_v", "friendly_a", "friendly_orientation", "enemy_p", "ball_p", "ball_v"):
            assert stored[key].dtype == np.float32, key
        assert stored["ts"].dtype == np.float64
    replay = load_columnar_replay(writer.path)
    assert replay.friendly_p.dtype == np.float64
    got = replay.frame_at(0)
    assert got.ts == 12.3456789
    assert isinstance(got.friendly_robots[1].p.x, float)  # np.float64 is a float; np.float32 is not
    assert got.friendly_robots[1].p.x == pytest.approx(1.23456789, abs=1e-6)
    assert got.referee.designated_position == pytest.approx((1.5, -0.25))


@pytest.mark.parametrize(
    "clock, stored",
    [
        (lambda ts: max(0.0, 1.0 - ts), [0]),  # the custom referee: stops at 0
        (lambda ts: 0.5 - ts, [0, 31]),  # a real referee into overtime: stored once, as it crosses 0
        (lambda ts: max(0.0, 1.0 - ts) if ts < 2.0 else 300.0 - (ts - 2.0), [0, 120]),  # a new stage
    ],
)
def test_a_stage_clock_that_stops_at_zero_is_not_stored_every_tick(tmp_path, clock, stored):
    """tournament_20261005_170958: the first-half stage clock reads 0 from 300 s on, and the
    sidecar stored a full referee message on every tick of the second 300 s of every match."""
    dt = 1.0 / 60
    writer = _make_writer(tmp_path)
    for i in range(180):
        ts = i * dt
        writer.write_frame(
            GameFrame(
                ts=ts,
                my_team_is_yellow=True,
                my_team_is_right=True,
                friendly_robots={1: _robot(1, 0, 0, True)},
                enemy_robots={},
                ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
                referee=dataclasses.replace(
                    _referee(RefereeCommand.NORMAL_START), time_sent=ts, time_received=ts, stage_time_left=clock(ts)
                ),
            )
        )
    writer.close()
    replay = load_columnar_replay(writer.path)

    assert sorted(replay.sparse_referee) == stored
    for i in range(180):
        assert replay.frame_at(i).referee.stage_time_left == pytest.approx(clock(i * dt), abs=1e-6)


def test_a_stage_clock_paused_in_a_stoppage_is_not_stored_every_tick(tmp_path):
    """The match clock stands still in STOP, HALT, kick-off and penalty preparation and ball
    placement (rulebook, Game Stages). Counting it down there made every stoppage tick a
    change: 11259 of 47166 ticks stored in a two-half match."""
    dt = 1.0 / 60
    writer = _make_writer(tmp_path)
    for i in range(120):
        ts = i * dt
        cmd = RefereeCommand.STOP if i < 60 else RefereeCommand.NORMAL_START
        clock = 100.0 - max(0, i - 60) * dt
        writer.write_frame(
            GameFrame(
                ts=ts,
                my_team_is_yellow=True,
                my_team_is_right=True,
                friendly_robots={1: _robot(1, 0, 0, True)},
                enemy_robots={},
                ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
                referee=dataclasses.replace(_referee(cmd), time_sent=ts, time_received=ts, stage_time_left=clock),
            )
        )
    writer.close()
    replay = load_columnar_replay(writer.path)

    assert sorted(replay.sparse_referee) == [0, 60]
    for i in range(120):
        assert replay.frame_at(i).referee.stage_time_left == pytest.approx(100.0 - max(0, i - 60) * dt)


# --- Changing ends at half-time -----------------------------------------------------------
# The side the recorded team defends was stored once per file, from the first tick, so a
# replay of a match where the teams change ends read the second half with the wrong goals.


def _side_frame(ts: float, is_right: bool) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=is_right,
        friendly_robots={1: _robot(1, 2.0, 1.0, True)},
        enemy_robots={2: _robot(2, -1.0, -0.5, False)},
        ball=Ball(p=Vector3D(1.5, 0.5, 0.0), v=Vector3D(1.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
        referee=None,
    )


def test_every_tick_reads_back_the_side_it_was_played_on(tmp_path):
    replay = _write(tmp_path, [_side_frame(0.0, True), _side_frame(0.1, True), _side_frame(0.2, False)])

    assert [replay.frame_at(i).my_team_is_right for i in range(3)] == [True, True, False]
    assert replay.my_team_is_right is True  # the first tick's, as before


def test_a_replay_from_before_the_side_was_stored_per_tick_keeps_one_side(tmp_path):
    import numpy as np

    replay = _write(tmp_path, [_side_frame(0.0, False), _side_frame(0.1, False)])
    path = tmp_path / "test_replay.npz"
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if k != "side_is_right"}
    np.savez(path, **arrays)

    old = load_columnar_replay(path)

    assert old.side_is_right is None
    assert [old.frame_at(i).my_team_is_right for i in range(2)] == [False, False]
    assert replay.frame_at(1).my_team_is_right is False


def test_a_scenario_from_the_second_half_is_turned_to_config_a_s_right_side(tmp_path):
    from utama_core.replay.scenario import scenario_from_replay

    _write(tmp_path, [_side_frame(0.0, True), _side_frame(1.0, False)])
    path = tmp_path / "test_replay.npz"

    first = scenario_from_replay(path, 0.0)
    second = scenario_from_replay(path, 1.0)

    assert (first.ball_x, first.ball_y, first.ball_vx) == pytest.approx((1.5, 0.5, 1.0))
    assert (second.ball_x, second.ball_y, second.ball_vx) == pytest.approx((-1.5, -0.5, -1.0))
    robot = second.friendly_robots[0]
    assert (robot.x, robot.y, robot.vx, robot.vy) == pytest.approx((-2.0, -1.0, -0.1, -0.2))
    assert robot.orientation == pytest.approx(1.5 - 3.141592653589793)
    assert (second.enemy_robots[0].x, second.enemy_robots[0].y) == pytest.approx((1.0, 0.5))
