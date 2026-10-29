"""`robot_calibration` recovers a robot's known numbers from a synthetic field log, and the log
it reads is what the radio and the referee really write."""

from __future__ import annotations

import math

import numpy as np
import pytest

from utama_core.analysis.robot_calibration import (
    BALL_LOST_M,
    FieldLog,
    calibrate,
    decode_feedback,
    decode_radio,
    dribbling,
    kick_bend,
    kick_direction_offset,
)
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.game_log_writer import RADIO_OUT, GameLogWriter
from utama_core.replay.ssl_log import REFEREE, read_records, referee_data
from utama_core.team_controller.src.controllers.real.real_robot_controller import (
    RealRobotController,
)
from utama_core.team_controller.src.generated_code.ssl_gc_referee_message_pb2 import (
    Referee,
)
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)

DELAY, ACCEL, TURN_ACCEL, GAIN = 0.08, 3.0, 30.0, 0.9
NOISE_M, NOISE_RAD, DROPPED = 0.001, math.radians(0.5), 0.02
FIRMWARE, VISION_ID = 3, 1


class _Serial:
    port = "fake"
    in_waiting = 0

    def __init__(self):
        self.written: list[bytes] = []

    def write(self, data):
        self.written.append(bytes(data))


def _radio_packet(controller: RealRobotController, serial: _Serial, command: RobotCommand) -> bytes:
    controller.add_robot_commands(command, VISION_ID)
    controller.send_robot_commands()
    return serial.written[-1]


def _vision_packet(t: float, x: float, y: float, th: float) -> bytes:
    packet = SSL_WrapperPacket()
    d = packet.detection
    d.frame_number, d.t_capture, d.t_sent, d.camera_id = int(t * 60), t, t, 0
    r = d.robots_yellow.add()
    r.confidence, r.robot_id, r.x, r.y, r.orientation, r.pixel_x, r.pixel_y = (
        1.0,
        VISION_ID,
        x * 1000,
        y * 1000,
        th,
        0,
        0,
    )
    return packet.SerializeToString()


def _script() -> list[tuple[float, RobotCommand]]:
    """(seconds, command): rest, then forward and back, sideways, a turn."""
    still = RobotCommand(0.0, 0.0, 0.0, False, False, False)
    moves = [(3.0, still)]
    for fwd, left, turn in [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, 2.0), (0, 0, -2.0)]:
        moves += [(1.5, RobotCommand(fwd, left, turn, False, False, False)), (2.0, still)]
    return moves


@pytest.fixture(scope="module")
def synthetic_log(tmp_path_factory):
    """A robot that follows each command `DELAY` late, at `GAIN` of it, accelerating at `ACCEL`
    (`TURN_ACCEL` turning), seen at 60 Hz with `NOISE_*` and `DROPPED` detections missing."""
    path = tmp_path_factory.mktemp("field") / "synthetic.log"
    log = GameLogWriter(path)
    log.run_info({"vision_to_cmd": {"yellow": {str(VISION_ID): FIRMWARE}, "blue": {}}})
    serial = _Serial()
    controller = RealRobotController(True, 1, vision_to_cmd_mapping={VISION_ID: FIRMWARE}, serial_port=serial)
    rng = np.random.default_rng(0)

    sent = []  # (time, command)
    t_end = 0.0
    for seconds, command in _script():
        for k in range(round(seconds * 60)):
            sent.append((t_end + k / 60, command))
        t_end += seconds
    for t, command in sent:
        log.write(RADIO_OUT, _radio_packet(controller, serial, command), ts_ns=int(t * 1e9))

    dt, x, y, th = 0.001, 0.0, 0.0, 0.3
    v = np.zeros(3)  # forward, left, turn
    next_frame = 0.008
    times = np.array([t for t, _ in sent])
    for i in range(int(t_end / dt)):
        t = i * dt
        j = np.searchsorted(times, t - DELAY, side="right") - 1
        target = GAIN * np.array(sent[j][1][:3]) if j >= 0 else np.zeros(3)
        limit = np.array([ACCEL, ACCEL, TURN_ACCEL]) * dt
        v += np.clip(target - v, -limit, limit)
        x += (v[0] * math.cos(th) - v[1] * math.sin(th)) * dt
        y += (v[0] * math.sin(th) + v[1] * math.cos(th)) * dt
        th += v[2] * dt
        if t >= next_frame:
            if rng.random() >= DROPPED:
                nx, ny, nth = rng.normal(0, [NOISE_M, NOISE_M, NOISE_RAD])
                log.write(4, _vision_packet(t, x + nx, y + ny, th + nth), ts_ns=int(t * 1e9))
            else:  # the camera sent the frame, without the robot
                packet = SSL_WrapperPacket()
                d = packet.detection
                d.frame_number, d.t_capture, d.t_sent, d.camera_id = int(t * 60), t, t, 0
                log.write(4, packet.SerializeToString(), ts_ns=int(t * 1e9))
            next_frame += 1 / 60
    log.close()
    return path


def test_the_writers_format_constants_are_the_readers():
    from utama_core.replay import game_log_writer, ssl_log

    assert (game_log_writer.LOG_HEADER, game_log_writer.REFEREE, game_log_writer.VISION_2014) == (
        ssl_log.LOG_HEADER,
        ssl_log.REFEREE,
        ssl_log.VISION_2014,
    )


def test_the_radio_decoder_reads_what_the_real_controller_sends():
    serial = _Serial()
    controller = RealRobotController(True, 1, vision_to_cmd_mapping={VISION_ID: FIRMWARE}, serial_port=serial)
    packet = _radio_packet(controller, serial, RobotCommand(0.5, -0.25, 2.0, kick=True, chip=False, dribble=False))
    ((fw, fwd, left, turn, kicker, dribbler),) = decode_radio(packet)
    assert (fw, fwd, left, turn, kicker, dribbler) == (FIRMWARE, 0.5, -0.25, 2.0, 0xF0, False)
    packet = _radio_packet(controller, serial, RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=True))
    assert decode_radio(packet)[0][5] is True


def test_the_feedback_decoder_reads_each_robots_ball_sensor():
    """Frames 0xAA, firmware id, length, data, 0x55; the sensor is bit 0 of the first data byte."""
    packet = bytes([0x00, 0xAA, 3, 2, 0x01, 0x7F, 0x55, 0xAA, 4, 1, 0x00, 0x55])
    assert decode_feedback(packet) == [(3, True), (4, False)]


@pytest.mark.parametrize("offset_deg", [7.0, -3.0])
def test_a_kicks_direction_is_measured_against_the_robots_heading(offset_deg):
    """2026-10-27: yellow's kicks left about 7 degrees off its heading."""
    heading = 0.4
    direction = heading + math.radians(offset_deg)
    t = np.arange(0.0, 0.5, 1 / 60)
    balls = np.column_stack([t, np.zeros_like(t), 2.0 * t * math.cos(direction), 2.0 * t * math.sin(direction)])
    assert kick_direction_offset(balls, 0.0, heading) == pytest.approx(offset_deg, abs=0.1)


def test_a_ball_that_barely_moved_gives_no_kick_direction():
    t = np.arange(0.0, 0.5, 1 / 60)
    balls = np.column_stack([t, np.zeros_like(t), 0.04 * t, np.zeros_like(t)])
    assert kick_direction_offset(balls, 0.0, 0.0) is None


@pytest.mark.parametrize("drift, kept", [(0.0, True), (BALL_LOST_M, False)])
def test_dribbling_reports_where_the_ball_sat_the_sensor_and_whether_it_stayed(drift, kept):
    """A robot facing +y turning 1 rad/s for 1 s with the dribbler on, the ball 0.09 m ahead and
    0.01 m to its left, drifting `drift` further ahead by the end; the sensor on for 3 of 4 readings."""
    t = np.arange(0.0, 1.0, 1 / 60)
    th = math.pi / 2 + t
    track = np.column_stack([t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), th])
    ahead = 0.09 + drift * t / t[-1]
    balls = np.column_stack(
        [t, np.zeros_like(t), ahead * np.cos(th) - 0.01 * np.sin(th), ahead * np.sin(th) + 0.01 * np.cos(th)]
    )
    commands = np.column_stack(
        [t, np.zeros_like(t), np.zeros_like(t), np.ones_like(t), np.zeros_like(t), np.ones_like(t)]
    )
    sensor = np.column_stack([t[::4], (np.arange(len(t[::4])) % 4 != 0).astype(float)])
    (row,) = dribbling(commands, track, balls, sensor)
    assert row["move"] == "turn 1 rad/s" and row["kept"] is kept
    assert row["ball_left_m"] == pytest.approx(0.01, abs=1e-3)
    assert row["sensor_on"] == pytest.approx(0.75, abs=0.03)
    if not drift:
        assert row["ball_forward_m"] == pytest.approx(0.09, abs=1e-3)


def test_calibration_recovers_the_synthetic_robots_numbers(synthetic_log):
    (result,) = calibrate(FieldLog(synthetic_log))
    assert result["robot"] == "yellow 1" and result["firmware_id"] == FIRMWARE
    # The onset, not the lag that lines the command up with the response: that one adds about
    # half the ramp (0.45 m/s / 3 m/s² / 2 = 75 ms) and would fail this.
    assert result["loop_delay_s"] == pytest.approx(DELAY, abs=0.01)
    forward, left, turn = (result["summary"][a] for a in ("forward", "left", "turn"))
    assert forward["steps"] == 2 and left["steps"] == 2 and turn["steps"] == 2
    for s in (forward, left, turn):
        assert s["gain"] == pytest.approx(GAIN, abs=0.02)
    assert forward["accel"] == pytest.approx(ACCEL, rel=0.15)
    # Turning reaches its speed in ~4 camera frames, so its acceleration is only roughly known.
    assert turn["accel"] == pytest.approx(TURN_ACCEL, rel=0.5)
    vision = result["vision"]
    assert vision["position_mm"] == pytest.approx(1000 * NOISE_M, rel=0.25)
    assert vision["heading_deg"] == pytest.approx(math.degrees(NOISE_RAD), rel=0.25)
    assert vision["missing_share"] == pytest.approx(DROPPED, abs=0.01)


def test_the_referee_record_reads_back_as_the_same_referee_data(tmp_path):
    data = RefereeData(
        source_identifier="utama",
        time_sent=12.5,
        time_received=12.5,
        referee_command=RefereeCommand.DIRECT_FREE_BLUE,
        referee_command_timestamp=10.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=200.0,
        blue_team=TeamInfo("blue", score=1, goalkeeper=0),
        yellow_team=TeamInfo("yellow", score=2, goalkeeper=1),
        designated_position=(1.5, -0.5),
        blue_team_on_positive_half=True,
        next_command=RefereeCommand.FORCE_START,
    )
    log = GameLogWriter(tmp_path / "r.log")
    log.referee(data)
    log.referee(data)  # unchanged: not written again
    log.close()
    records = [p for _, kind, p in read_records(tmp_path / "r.log") if kind == REFEREE]
    assert len(records) == 1
    msg = Referee()
    msg.ParseFromString(records[0])
    back = referee_data(msg, received_s=12.5)
    assert back.referee_command == data.referee_command and back.stage == data.stage
    assert (back.yellow_team.score, back.blue_team.score) == (2, 1)
    assert back.designated_position == pytest.approx(data.designated_position)
    assert back.blue_team_on_positive_half is True and back.next_command == RefereeCommand.FORCE_START


def _kick(left_cm: float, offset_deg: float) -> dict:
    return {"chip": False, "ball_left_cm": left_cm, "direction_offset_deg": offset_deg}


def test_the_kick_bend_is_fitted_through_zero_per_cm_off_centre():
    kicks = [_kick(x, 4.5 * x) for x in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)]
    kicks += [
        _kick(5.0, 90.0),
        _kick(1.0, None),
        {**_kick(1.0, 40.0), "chip": True},
    ]  # not on the dribbler, unmeasured, a chip

    bend = kick_bend(kicks)

    assert bend["deg_per_cm"] == pytest.approx(4.5)
    assert bend["kicks"] == 6 and bend["rms_deg"] == 0.0


def test_too_few_kicks_give_no_bend():
    assert kick_bend([_kick(1.0, 4.0)] * 4) is None


def test_correcting_a_tag_moves_that_robots_detections_onto_the_robot():
    log = FieldLog.__new__(FieldLog)
    # (t, camera, x, y, heading): facing +y, the tag read 3.5 deg clockwise and 1 cm ahead.
    log.robots = {
        ("yellow", 0): np.array([[0.0, 0, 1.0, 1.0, math.pi / 2]]),
        ("blue", 0): np.array([[0.0, 0, 0.0, 0.0, 0.0]]),
    }

    log.correct_tags({("yellow", 0): (0.01, 0.0, math.radians(-3.5))})

    assert log.robots[("yellow", 0)][0, 2:] == pytest.approx([1.0, 0.99, math.pi / 2 + math.radians(3.5)])
    assert log.robots[("blue", 0)][0, 2:].tolist() == [0.0, 0.0, 0.0]
