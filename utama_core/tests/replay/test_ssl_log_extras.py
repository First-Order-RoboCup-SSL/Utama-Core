"""Tests for the official-log converter's sidecars (`replay/ssl_log.py`): the tracker's kicks,
visibility and angular velocity, raw camera detections, and adding them to a replay converted
before they existed (`tools/convert_ssl_log.py --backfill-extras`)."""

from __future__ import annotations

import gzip
import json
import struct

import numpy as np
import pytest

from tools import convert_ssl_log
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.ssl_log import (
    LOG_HEADER,
    REFEREE,
    TRACKER,
    VISION_2014,
    _add_kick,
    _kick_rows,
    _VisionRows,
    convert_log,
)
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)
from utama_core.team_controller.src.generated_code.ssl_vision_detection_tracked_pb2 import (
    KickedBall,
)
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_tracked_pb2 import (
    TrackerWrapperPacket,
)

S = 1_000_000_000  # ns
T0 = 1_700_000_000 * S
TRACKER_LAG_S = 0.02  # tracked frames arrive this long after their tracker timestamp


def _write_log(path, records) -> None:
    body = b"".join(struct.pack(">qii", t, kind, len(payload)) + payload for t, kind, payload in records)
    path.write_bytes(gzip.compress(LOG_HEADER + struct.pack(">i", 1) + body))


def _referee(ts_ns, stage, command, counter) -> tuple:
    msg = Referee(
        packet_timestamp=ts_ns // 1000,
        stage=stage.value,
        command=command.value,
        command_counter=counter,
        command_timestamp=ts_ns // 1000,
        blue_team_on_positive_half=True,
    )
    for team, name in ((msg.yellow, "Yellows"), (msg.blue, "Blues")):
        team.name, team.score, team.goalkeeper = name, 0, 0
        team.red_cards = team.yellow_cards = team.timeouts = team.timeout_time = 0
    return ts_ns, REFEREE, msg.SerializeToString()


def _tracked(ts_ns, visibility, kick=None, source="TIGERs") -> tuple:
    """Yellow 7 and blue 12 on the field; `kick`: (start s after T0, team, id, speed);
    `visibility` None sends none."""
    packet = TrackerWrapperPacket(uuid=source, source_name=source)
    frame = packet.tracked_frame
    frame.frame_number, frame.timestamp = 0, ts_ns / S - TRACKER_LAG_S
    shown = {} if visibility is None else {"visibility": visibility}
    frame.balls.add(pos=dict(x=0.0, y=0.0, z=0.0), **shown)
    frame.robots.add(robot_id=dict(id=7, team=1), pos=dict(x=1.0, y=0.5), orientation=0.0, vel_angular=2.0)
    shown = {} if visibility is None else {"visibility": visibility / 2}
    frame.robots.add(robot_id=dict(id=12, team=2), pos=dict(x=-1.0, y=0.5), orientation=0.0, **shown)
    if kick is not None:
        start_s, team, rid, speed = kick
        frame.kicked_ball.pos.x, frame.kicked_ball.pos.y = 1.0, 0.5
        frame.kicked_ball.vel.x, frame.kicked_ball.vel.y, frame.kicked_ball.vel.z = speed, 0.0, 0.0
        frame.kicked_ball.start_timestamp = T0 / S + start_s
        frame.kicked_ball.robot_id.id, frame.kicked_ball.robot_id.team = rid, team
    return ts_ns, TRACKER, packet.SerializeToString()


def _camera(ts_ns, camera, ball_mm, robot_mm) -> tuple:
    packet = SSL_WrapperPacket()
    d = packet.detection
    d.frame_number, d.t_capture, d.t_sent, d.camera_id = 5, ts_ns / S - 0.01, ts_ns / S, camera
    d.balls.add(confidence=0.9, x=ball_mm[0], y=ball_mm[1], z=0.0, pixel_x=10.0, pixel_y=20.0, area=12)
    d.robots_yellow.add(confidence=0.8, robot_id=7, x=robot_mm[0], y=robot_mm[1], orientation=0.3, pixel_x=1, pixel_y=2)
    return ts_ns, VISION_2014, packet.SerializeToString()


def _kick_of(t_s) -> tuple | None:
    """The kick a tracked frame at `t_s` reports: yellow 7 kicks at 0.5 s; the tracker gives it to
    blue 12 from 0.6 s and refines the speed until 0.9 s; yellow 7 kicks again at 1.2 s."""
    if 0.5 <= t_s < 1.0:
        return (0.5 + (0.01 if t_s >= 0.6 else 0.0), 1 if t_s < 0.6 else 2, 7 if t_s < 0.6 else 12, min(t_s, 0.9) * 5)
    if 1.2 <= t_s < 1.4:
        return (1.2, 1, 7, 3.0)
    return None


@pytest.fixture
def log(tmp_path):
    """2 s of first half, then the post-game stage; tracked frames at 100 Hz (faster than the
    replay's 60 Hz ticks), ball visibility rising from 0 to 1, and a camera frame every 0.5 s from
    before the first referee message to after the last."""
    half = Stage.NORMAL_FIRST_HALF
    records = [
        _referee(T0, half, RefereeCommand.NORMAL_START, 1),
        _referee(T0 + 2 * S, Stage.POST_GAME, RefereeCommand.HALT, 2),
    ]
    for k in range(0, 201):
        t_s = k / 100
        records.append(_tracked(T0 + k * S // 100, visibility=t_s / 2, kick=_kick_of(t_s)))
    for k in range(-1, 6):
        records.append(_camera(T0 + k * S // 2, camera=k % 2, ball_mm=(1000.0 * k, -500.0), robot_mm=(250.0, 750.0)))
    records.sort(key=lambda r: r[0])
    path = tmp_path / "2026-07-01_Yellows-vs-Blues.log.gz"
    _write_log(path, records)
    return path


def test_a_kick_the_tracker_repeats_and_reassigns_is_one_row_with_its_last_kicker_and_estimate(log, tmp_path):
    summary = convert_log(log, tmp_path / "out")
    kicks = json.loads((tmp_path / "out" / f"{log.name[:-7]}.kicks.json").read_text())

    assert summary["extras"] == {
        "kicks": 2,
        "kicks_source": "TIGERs",
        "visibility_source": "TIGERs",
        "tracker": True,
        "raw_vision": False,
    }
    first, second = kicks
    # start on the replay clock: the first report's start time plus its receive-minus-tracker lag
    assert first["ts"] == pytest.approx(0.5 + TRACKER_LAG_S, abs=1e-3)
    assert (first["team"], first["robot_id"], first["kickers"]) == ("blue", 12, 2)
    assert first["vel_first"][0] == pytest.approx(2.5) and first["speed"] == pytest.approx(4.5)
    # one report per tick (60 Hz) whose newest tracked frame (100 Hz) is new: ticks 30-59
    assert first["reports"] == 30
    assert (second["team"], second["robot_id"], second["kickers"], second["reports"]) == ("yellow", 7, 1, 12)


def test_kicks_and_visibility_come_from_another_source_when_the_replays_tracker_sends_none(tmp_path):
    # as 2026: the busiest tracker (ER-FORCE, the replay's) sends neither; TIGERs sends both
    half = Stage.NORMAL_FIRST_HALF
    records = [
        _referee(T0, half, RefereeCommand.NORMAL_START, 1),
        _referee(T0 + 2 * S, Stage.POST_GAME, RefereeCommand.HALT, 2),
    ]
    for k in range(0, 201):
        records.append(_tracked(T0 + k * S // 100, visibility=None, source="ER-FORCE"))
        if k % 2 == 0:
            records.append(_tracked(T0 + k * S // 100, visibility=0.75, kick=_kick_of(k / 100)))
    records.sort(key=lambda r: r[0])
    path = tmp_path / "two-trackers.log.gz"
    _write_log(path, records)
    summary = convert_log(path, tmp_path / "out")
    extra = np.load(tmp_path / "out" / "two-trackers.tracker.zip")

    assert summary["world_source"] == "ER-FORCE"
    assert (summary["extras"]["kicks_source"], summary["extras"]["visibility_source"]) == ("TIGERs", "TIGERs")
    assert summary["extras"]["kicks"] == 2
    assert extra["ball_visibility"][30] == pytest.approx(0.75)
    assert extra["robot_vel_angular"][30, 0, 7] == pytest.approx(2.0)  # from ER-FORCE's own frames


def test_kick_starts_more_than_the_same_kick_window_apart_are_two_kicks():
    kicks: list[dict] = []
    for start in (10.0, 10.04, 10.1):
        kick = KickedBall(start_timestamp=start)
        kick.pos.x = kick.pos.y = 0.0
        kick.vel.x = kick.vel.y = kick.vel.z = 0.0
        kick.robot_id.id, kick.robot_id.team = 3, 1
        _add_kick(kicks, kick, recv_ns=int((start + 0.5) * S), frame_timestamp=start + 0.5)
    assert [k["reports"] for k in kicks] == [2, 1]
    assert kicks[0]["start_ns"] == int(10.0 * S)


def test_a_kick_after_a_colour_swap_is_given_in_kick_off_colours_and_one_outside_the_window_dropped():
    tick_ns = np.arange(4) * S
    swapped = np.array([False, False, True, True])
    kicks = [
        {
            "start_ns": t * S,
            "kicker": (0, 1),
            "kickers": {(0, 1)},
            "pos": [0, 0],
            "vel_first": [1, 0, 0],
            "vel": [1, 0, 0],
            "reports": 1,
        }
        for t in (1, 2, 5)
    ]
    rows = _kick_rows(kicks, start_ns=0, end_ns=4 * S, tick_ns=tick_ns, swapped=swapped)
    assert [(r["ts"], r["team"]) for r in rows] == [(1.0, "yellow"), (2.0, "blue")]


def test_visibility_and_angular_velocity_land_on_the_replays_ticks(log, tmp_path):
    convert_log(log, tmp_path / "out")
    extra = np.load(tmp_path / "out" / f"{log.name[:-7]}.tracker.zip")

    ticks = len(extra["ts"])
    assert ticks == 120 and extra["robot_visibility"].shape == (ticks, 2, 16)
    # tick 30 (0.5 s) takes the newest tracked frame at or before it: the one at 0.5 s
    assert extra["ball_visibility"][30] == pytest.approx(0.25)
    assert extra["robot_visibility"][30, 1, 12] == pytest.approx(0.125)
    assert np.isnan(extra["robot_visibility"][30, 0, 7])  # yellow 7's visibility is not given
    assert extra["robot_vel_angular"][30, 0, 7] == pytest.approx(2.0)
    assert np.isnan(extra["robot_visibility"][30, 0, 3])  # no such robot


def test_raw_camera_detections_inside_the_window_round_trip_in_metres(log, tmp_path):
    summary = convert_log(log, tmp_path / "out", raw_vision=True)
    vision = np.load(tmp_path / "out" / f"{log.name[:-7]}.vision.zip")

    assert summary["extras"]["raw_vision"]
    # camera frames at -0.5 s and 2.0 s and after are outside the window [0, 2) s
    np.testing.assert_allclose(vision["frame_ts"], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(vision["frame_camera"], [0, 1, 0, 1])
    np.testing.assert_allclose(vision["frame_t_sent"] - vision["frame_t_capture"], 0.01, atol=1e-6)
    np.testing.assert_array_equal(vision["ball_frame"], [0, 1, 2, 3])
    np.testing.assert_allclose(vision["ball_x"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(vision["ball_y"], -0.5)
    np.testing.assert_array_equal(vision["ball_area"], 12)
    np.testing.assert_allclose(vision["robot_x"], 0.25)
    np.testing.assert_array_equal(vision["robot_id"], 7)
    np.testing.assert_array_equal(vision["robot_team"], 0)
    assert np.isnan(vision["robot_height"]).all()  # not sent


def test_raw_detections_after_a_colour_swap_go_back_to_kick_off_colours():
    rows = _VisionRows()
    for k in range(3):
        packet = SSL_WrapperPacket()
        packet.detection.frame_number, packet.detection.camera_id = k, 0
        packet.detection.t_capture = packet.detection.t_sent = float(k)
        packet.detection.robots_blue.add(confidence=1.0, robot_id=2, x=0.0, y=0.0, pixel_x=0, pixel_y=0)
        rows.add(k * S, packet.detection)
    out = rows.arrays(0, 3 * S, np.arange(3) * S, swapped=np.array([False, True, True]))
    np.testing.assert_array_equal(out["robot_team"], [1, 0, 0])


def test_backfill_adds_the_sidecars_without_touching_the_replay(log, tmp_path, monkeypatch):
    out = tmp_path / "out"
    convert_log(log, out)
    stem = log.name[:-7]
    replay_bytes = (out / f"{stem}.npz").read_bytes()
    for suffix in (".kicks.json", ".tracker.zip"):
        (out / f"{stem}{suffix}").unlink()
    source = json.loads((out / f"{stem}.source.json").read_text())
    del source["extras"]  # as converted before the sidecars existed
    (out / f"{stem}.source.json").write_text(json.dumps(source))
    assert not convert_ssl_log.has_extras(out, stem, raw_vision=False)

    monkeypatch.setattr(
        convert_ssl_log, "_download", lambda url, path, _expected_bytes=0: path.write_bytes(log.read_bytes())
    )
    job = {
        "name": log.name,
        "year": 2026,
        "bytes": 0,
        "url": "",
        "out": str(tmp_path),
        "raw_dir": str(tmp_path / "raw"),
    }
    (tmp_path / "raw").mkdir()
    (tmp_path / "2026").symlink_to(out)
    skipped = convert_ssl_log._corpus_job({**job, "raw_vision": True, "backfill_extras": False})
    row = convert_ssl_log._corpus_job({**job, "raw_vision": True, "backfill_extras": True})
    again = convert_ssl_log._corpus_job({**job, "raw_vision": True, "backfill_extras": True})

    assert (skipped["status"], row["status"], again["status"]) == ("exists", "extras added", "exists")
    assert (out / f"{stem}.npz").read_bytes() == replay_bytes
    assert (out / f"{stem}.kicks.json").exists() and (out / f"{stem}.vision.zip").exists()
    assert json.loads((out / f"{stem}.source.json").read_text())["extras"]["raw_vision"]
    assert not (tmp_path / "raw" / log.name).exists()  # the download is deleted


def test_backfill_refuses_a_replay_of_another_length(log, tmp_path):
    out = tmp_path / "out"
    convert_log(log, out)
    source_path = out / f"{log.name[:-7]}.source.json"
    source = json.loads(source_path.read_text())
    source["duration_s"] += 1.0
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="gives"):
        convert_log(log, out, write_replay=False)
