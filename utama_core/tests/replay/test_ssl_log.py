"""Tests for converting official SSL game logs (`replay/ssl_log.py`), on small synthetic logs:
the record reader, and how robots, sides, units and commands land in the converted replay."""

from __future__ import annotations

import gzip
import json
import struct

import numpy as np
import pytest

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.columnar_reader import load_columnar_replay
from utama_core.replay.ssl_log import (
    LOG_HEADER,
    REFEREE,
    TRACKER,
    VISION_2014,
    _Scan,
    _sides_from_goalkeepers,
    _World,
    convert_log,
    free_kick_taken,
    log_stem,
    read_records,
)
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_tracked_pb2 import (
    TrackerWrapperPacket,
)

S = 1_000_000_000  # ns
T0 = 1_700_000_000 * S
DIRECT_FREE_BLUE = RefereeCommand.DIRECT_FREE_BLUE.value
STOP = RefereeCommand.STOP.value


def _record(ts_ns: int, msg_type: int, payload: bytes) -> bytes:
    return struct.pack(">qii", ts_ns, msg_type, len(payload)) + payload


def _write_log(path, records) -> None:
    data = LOG_HEADER + struct.pack(">i", 1) + b"".join(_record(*r) for r in records)
    path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)


def _referee(ts_ns, stage, command, counter, score=(0, 0), blue_on_positive=None, **fields) -> tuple:
    msg = Referee(
        packet_timestamp=ts_ns // 1000,
        stage=stage.value,
        command=command.value,
        command_counter=counter,
        command_timestamp=ts_ns // 1000,
        **fields,
    )
    for team, name, goals in ((msg.yellow, "Yellows", score[0]), (msg.blue, "Blues", score[1])):
        team.name, team.score, team.goalkeeper = name, goals, 0
        team.red_cards = team.yellow_cards = team.timeouts = team.timeout_time = 0
    if blue_on_positive is not None:
        msg.blue_team_on_positive_half = blue_on_positive
    return ts_ns, REFEREE, msg.SerializeToString()


def _tracked(ts_ns, robots, ball=(0.0, 0.0), source="TIGERs") -> tuple:
    """`robots`: (team, id, x, y) with team 1 yellow, 2 blue, positions in metres."""
    packet = TrackerWrapperPacket(uuid=source, source_name=source)
    frame = packet.tracked_frame
    frame.frame_number, frame.timestamp = 0, ts_ns / S
    frame.balls.add(pos=dict(x=ball[0], y=ball[1], z=0.0))
    for team, rid, x, y in robots:
        frame.robots.add(robot_id=dict(id=rid, team=team), pos=dict(x=x, y=y), orientation=0.5)
    return ts_ns, TRACKER, packet.SerializeToString()


def _geometry(ts_ns, length_mm, width_mm) -> tuple:
    packet = SSL_WrapperPacket()
    field = packet.geometry.field
    field.field_length, field.field_width = length_mm, width_mm
    field.goal_width, field.goal_depth, field.boundary_width = 1800, 180, 300
    return ts_ns, VISION_2014, packet.SerializeToString()


def test_read_records_reads_plain_and_gzipped_logs_and_drops_a_cut_off_record(tmp_path):
    records = [(T0, REFEREE, b"abc"), (T0 + S, TRACKER, b"")]
    for name in ("match.log", "match.log.gz"):
        path = tmp_path / name
        data = LOG_HEADER + struct.pack(">i", 1) + b"".join(_record(*r) for r in records)
        data += _record(T0 + 2 * S, REFEREE, b"0123456789")[:-1]  # the logger stopped mid-record
        path.write_bytes(gzip.compress(data) if name.endswith(".gz") else data)
        assert list(read_records(path)) == records


def test_read_records_rejects_a_file_that_is_not_a_game_log(tmp_path):
    path = tmp_path / "match.log"
    path.write_bytes(b"NOT_A_LOG_FILE\x00\x00\x00\x01")
    with pytest.raises(ValueError, match="not an SSL game log"):
        list(read_records(path))


def test_a_published_name_is_unescaped_to_its_stem():
    assert log_stem("2023-07-07_Rob%C3%B4Cin-vs-TIGERs.log.gz") == "2023-07-07_RobôCin-vs-TIGERs"
    assert log_stem("2023-07-07_Rob%25C3%25B4Cin-vs-TIGERs.log.gz") == "2023-07-07_RobôCin-vs-TIGERs"
    assert log_stem("match.log") == "match"


def test_a_free_kick_reads_taken_from_the_tick_its_ball_has_moved_five_centimetres():
    command = np.array([STOP] + [DIRECT_FREE_BLUE] * 5 + [STOP])
    counter = np.array([1, 2, 2, 2, 2, 2, 3])
    x = np.array([9.0, 0.0, 0.0, 0.049, 0.05, 2.0, 3.0])
    ball = np.stack([x, np.zeros_like(x)], axis=1)
    np.testing.assert_array_equal(free_kick_taken(command, counter, ball), [0, 0, 0, 0, 1, 1, 0])


def test_a_free_kick_whose_ball_never_moves_is_never_taken_and_a_new_counter_restarts_the_check():
    # two free kicks in a row: the second is measured from where its own ball stood
    command = np.array([DIRECT_FREE_BLUE] * 6)
    counter = np.array([4, 4, 4, 5, 5, 5])
    x = np.array([0.0, 0.0, 0.04, 2.0, 2.0, 2.04])
    ball = np.stack([x, np.zeros_like(x)], axis=1)
    assert not free_kick_taken(command, counter, ball).any()


def test_sides_without_the_referee_field_follow_yellows_goalkeeper_per_stage():
    world = _World(4)
    stage = np.array([Stage.NORMAL_FIRST_HALF.value] * 2 + [Stage.NORMAL_SECOND_HALF.value] * 2)
    keepers = np.array([2, 2, 2, 2])
    world.robot_p[:2, 0, 2] = (4.0, 0.0)  # yellow's keeper at the +x goal, then at the -x goal
    world.robot_p[2:, 0, 2] = (-4.0, 0.0)
    np.testing.assert_array_equal(_sides_from_goalkeepers(world, stage, keepers), [True, True, False, False])


@pytest.fixture
def converted(tmp_path):
    """A 2 s match: first half for 1 s (blue on +x), second half for 1 s (blue on -x), then the
    post-game stage. Yellow 3 leaves after 0.5 s, yellow 7 plays throughout, blue 12 comes on
    at 1 s and a robot with id 16 is out of the pattern set. A second, sparser tracker source
    puts everything elsewhere. The free kick at 0.25 s is taken at 0.5 s; yellow scores at 1 s."""
    half, second = Stage.NORMAL_FIRST_HALF, Stage.NORMAL_SECOND_HALF
    records = [
        _referee(T0, half, RefereeCommand.STOP, 1, blue_on_positive=True, stage_time_left=300_000_000),
        _referee(T0 + S // 100, half, RefereeCommand.STOP, 1, blue_on_positive=True, stage_time_left=299_990_000),
        _referee(
            T0 + S // 8,
            half,
            RefereeCommand.BALL_PLACEMENT_BLUE,
            2,
            blue_on_positive=True,
            designated_position=dict(x=1500.0, y=-1000.0),
        ),
        _referee(T0 + S // 4, half, RefereeCommand.DIRECT_FREE_BLUE, 3, blue_on_positive=True),
        _referee(T0 + S, half, RefereeCommand.STOP, 4, score=(1, 0), blue_on_positive=True),
        _referee(T0 + S + S // 60, second, RefereeCommand.STOP, 4, score=(1, 0), blue_on_positive=False),
        _referee(T0 + 2 * S, Stage.POST_GAME, RefereeCommand.HALT, 5, score=(1, 0), blue_on_positive=False),
        _geometry(T0, 12000, 9000),
    ]
    for k in range(0, 2 * 60 + 1):
        t = T0 + k * S // 60
        robots = (
            [(1, 7, 1.0, 0.5), (1, 16, 0.0, 0.0)] + [(1, 3, -1.0, 0.0)] * (k < 30) + [(2, 12, 2.0, 1.0)] * (k >= 60)
        )
        records.append(_tracked(t, robots, ball=(0.0 if k < 30 else 0.5, 0.0)))
        if k % 2 == 0:
            records.append(_tracked(t, [(1, 5, -3.0, -3.0)], source="ER-Force"))
    records.sort(key=lambda r: r[0])
    path = tmp_path / "2026-07-01_Yellows-vs-Blues.log.gz"
    _write_log(path, records)
    summary = convert_log(path, tmp_path / "out")
    return load_columnar_replay(tmp_path / "out" / "2026-07-01_Yellows-vs-Blues.npz"), summary, path


def test_a_resent_referee_message_is_kept_once(converted):
    # the game controller resends its state; only a change of something but the clocks is kept
    *_, path = converted
    assert [msg.command_counter for _, msg in _Scan(path).referee] == [1, 2, 3, 4, 4, 5]


def test_yellow_is_friendly_with_one_column_per_robot_id_it_used(converted):
    replay, summary, _ = converted
    assert replay.my_team_is_yellow
    assert replay.friendly_ids.tolist() == [3, 7]
    assert replay.enemy_ids.tolist() == [12]
    assert summary["world_source"] == "TIGERs"  # the source with the most frames
    assert summary["robot_ids_dropped"] > 0


def test_a_robot_is_nan_off_the_field_and_another_id_comes_on_without_losing_either(converted):
    replay, *_ = converted
    three, seven = 0, 1
    # yellow 3 is in the frames of ticks 0-29, blue 12 from tick 60 on
    assert not np.isnan(replay.friendly_p[:30, three, 0]).any()
    assert np.isnan(replay.friendly_p[30:, three, 0]).all()
    assert not np.isnan(replay.friendly_p[:, seven, 0]).any()
    assert np.isnan(replay.enemy_p[:60, 0, 0]).all()
    assert not np.isnan(replay.enemy_p[60:, 0, 0]).any()
    np.testing.assert_allclose(replay.friendly_p[0, seven], [1.0, 0.5], atol=1e-6)


def test_the_side_follows_blue_team_on_positive_half_per_tick(converted):
    replay, *_ = converted
    first_half = replay.stage == Stage.NORMAL_FIRST_HALF.value
    second_half = replay.stage == Stage.NORMAL_SECOND_HALF.value
    assert first_half.any() and second_half.any()
    assert not replay.side_is_right[first_half].any()  # blue defends +x, so yellow defends -x
    assert replay.side_is_right[second_half].all()


def test_referee_units_scores_and_the_window_end_at_the_post_game_stage(converted):
    replay, summary, _ = converted
    assert replay.ts[-1] < 2.0 and replay.n_ticks == 120
    first = replay.frame_at(0).referee
    assert first.stage_time_left == pytest.approx(300.0, abs=0.05)
    placing = np.flatnonzero(replay.referee_command == RefereeCommand.BALL_PLACEMENT_BLUE.value)
    np.testing.assert_allclose(replay.designated_position[placing[0]], [1.5, -1.0])
    assert replay.frame_at(replay.n_ticks - 1).referee.yellow_team.score == 1
    assert summary["score"] == {"yellow": 1, "blue": 0}
    assert summary["field"]["field_length"] == 12.0


def test_a_free_kick_reads_normal_start_once_its_ball_is_in_play(converted):
    replay, *_ = converted
    command = replay.referee_command
    # given at tick 15, its ball moves at tick 30, stopped at tick 60
    assert (command[15:30] == RefereeCommand.DIRECT_FREE_BLUE.value).all()
    assert (command[30:60] == RefereeCommand.NORMAL_START.value).all()
    assert command[60] == RefereeCommand.STOP.value


def test_the_source_sidecar_names_the_log_and_teams(converted, tmp_path):
    source = json.loads((tmp_path / "out" / "2026-07-01_Yellows-vs-Blues.source.json").read_text())
    assert (source["yellow"], source["blue"]) == ("Yellows", "Blues")
    assert source["tracker_frames"] == {"TIGERs": 121, "ER-Force": 61}
