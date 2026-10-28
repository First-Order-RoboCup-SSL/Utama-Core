"""A robot's numbers from a recorded field test: what the sim and the controller need to know.

Reads an SSL game log written by `GameLogWriter` (`StrategyRunner(game_log_path=...)`) on real
robots: the radio packets as sent and the camera's raw detections, on one clock. Best from a
`--calibrate` run of `examples/shooting_drill.py` (fixed moves, `ScriptedMovesTactic`);
any run works, with fewer clean steps.

Per robot that was sent commands:
- **Steps**: each constant command held at least `MIN_STEP_S`, with the delay, acceleration
  and speed reached of the sim's model of a move (`ramp`: still, then constant acceleration up
  to a speed) fitted to the robot's positions, and the speed's ratio to the command (`gain`).
- **Loop delay**: the median step delay. Command sent -> robot moves -> camera sees it ->
  packet received: the whole sensing-to-action loop, the sim's latency. Measured at the
  onset, not by lining the command up with the response, which would add half the time
  spent accelerating (the sim models that separately).
- **Vision noise** while the robot is at rest (`STILL_AFTER_S` after its last move): the
  standard deviation of a single detection, from second differences / sqrt(6) (as
  `physics_ranges` measures the league's), and the share of the camera's frames missing it.
  The same for a ball at rest.
- **Kicks**: per kick sent, the ball's top speed within 0.6 s, when the ball was at the robot, and
  the direction it left in against the robot's heading (`direction_offset_deg`, anticlockwise +).
- **Dribbling**: per constant command held at least `DRIBBLE_MIN_S` with the dribbler on, where
  the ball sat in the robot's frame (forward of its centre, to its left), how much of the time the
  ball sensor read on, and whether the ball was still on at the end (`BALL_LOST_M`). A
  `examples/calibrate_robot.py` run holds, turns and drives with the ball, then kicks it 3 times.

    pixi run python tools/robot_calibration.py replays/field/<run>.log [--json OUT]
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from utama_core.replay.game_log_writer import RADIO_IN, RADIO_OUT, RUN_INFO
from utama_core.replay.ssl_log import VISION_2014, read_records
from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)

FRAME_BYTES = 12  # 0xAA, firmware id, fwd/left/turn float16 LE, dribbler 2 bytes, kicker, 0x55
MAX_LAG_S = 0.5
LAG_STEP_S = 0.005
MIN_STEP_S = 0.4
STILL_AFTER_S = 1.0
DIFF = 3  # velocity: central difference over +-3 detections (0.1 s at 60 Hz)
KICK_WINDOW_S = 0.6
KICK_REACH_M = 0.3  # the ball within this of the robot when the kick is sent
AXES = ("forward", "left", "turn")
DRIBBLE_MIN_S = 0.5
BALL_LOST_M = 0.15  # ball centre this far from the robot's: off the dribbler (held, ~0.09 m)
KICK_DIRECTION_S = (0.05, 0.4)  # the ball's travel after a kick, for its direction


def decode_radio(packet: bytes) -> list[tuple[int, float, float, float, int, bool]]:
    """(firmware id, forward, left, turn, kicker byte, dribbler on) per robot frame in one radio packet."""
    out = []
    for i in range(0, len(packet) - FRAME_BYTES + 1, FRAME_BYTES):
        fr = packet[i : i + FRAME_BYTES]
        if fr[0] != 0xAA or fr[-1] != 0x55 or fr[1] == 0xFF:
            continue
        fwd, left, turn = np.frombuffer(bytes(fr[2:8]), dtype="<f2").astype(float)
        out.append((fr[1], float(fwd), float(left), float(turn), fr[10], bool(fr[8] or fr[9])))
    return out


def decode_feedback(packet: bytes) -> list[tuple[int, bool]]:
    """(firmware id, ball sensor on) per robot frame in one radio feedback packet: frames are
    0xAA, firmware id, length, data (sensor in bit 0 of its first byte), 0x55."""
    out, i = [], 0
    while i + 4 < len(packet):
        n = packet[i + 2]
        if packet[i] == 0xAA and i + 3 + n < len(packet) and packet[i + 3 + n] == 0x55 and n > 0:
            out.append((packet[i + 1], bool(packet[i + 3] & 1)))
            i += 4 + n
        else:
            i += 1
    return out


class FieldLog:
    """The parts of a game log this module uses, times in seconds from the first record."""

    def __init__(self, path):
        self.info: dict = {}
        commands = defaultdict(list)  # firmware id -> [(t, fwd, left, turn, kicker, dribbler)]
        sensor = defaultdict(list)  # firmware id -> [(t, ball sensor on)]
        robots = defaultdict(list)  # (colour, vision id) -> [(t, camera, x, y, th)]
        balls = []  # (t, camera, x, y)
        self.camera_frames: dict[int, list[float]] = defaultdict(list)
        t0 = None
        packet = SSL_WrapperPacket()
        for ts_ns, msg_type, payload in read_records(path):
            t0 = ts_ns if t0 is None else t0
            t = (ts_ns - t0) / 1e9
            if msg_type == RUN_INFO:
                self.info = json.loads(payload)
            elif msg_type == RADIO_OUT:
                for fw, fwd, left, turn, kicker, dribbler in decode_radio(payload):
                    commands[fw].append((t, fwd, left, turn, kicker, dribbler))
            elif msg_type == RADIO_IN:
                for fw, on in decode_feedback(payload):
                    sensor[fw].append((t, on))
            elif msg_type == VISION_2014:
                packet.Clear()
                packet.ParseFromString(payload)
                if not packet.HasField("detection"):
                    continue
                d = packet.detection
                self.camera_frames[d.camera_id].append(t)
                for colour, detected in (("yellow", d.robots_yellow), ("blue", d.robots_blue)):
                    for r in detected:
                        robots[(colour, r.robot_id)].append((t, d.camera_id, r.x / 1000, r.y / 1000, r.orientation))
                if d.balls:
                    b = max(d.balls, key=lambda b: b.confidence)
                    balls.append((t, d.camera_id, b.x / 1000, b.y / 1000))
        self.commands = {fw: np.array(rows, dtype=float) for fw, rows in commands.items()}
        self.sensor = {fw: np.array(rows, dtype=float) for fw, rows in sensor.items()}
        self.robots = {key: np.array(rows) for key, rows in robots.items()}
        self.balls = np.array(balls).reshape(-1, 4)

    def robot_of(self, firmware_id: int) -> Optional[tuple[str, int]]:
        """(colour, vision id) driven through `firmware_id`, from the run info."""
        for colour, mapping in self.info.get("vision_to_cmd", {}).items():
            for vision_id, fw in mapping.items():
                if fw == firmware_id:
                    return colour, int(vision_id)
        return None


def _main_camera(rows: np.ndarray) -> np.ndarray:
    """`rows` from the camera that saw the most of them, in time order, one per time."""
    cam = Counter(rows[:, 1].astype(int).tolist()).most_common(1)[0][0]
    rows = rows[rows[:, 1] == cam]
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    return rows[np.r_[True, np.diff(rows[:, 0]) > 0]]


def local_velocity(track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Times and (forward, left, turn) velocity in the robot's frame from (t, cam, x, y, th) rows."""
    t, x, y, th = track[:, 0], track[:, 2], track[:, 3], np.unwrap(track[:, 4])
    k = DIFF
    if len(t) <= 2 * k:
        return np.empty(0), np.empty((0, 3))
    dt = t[2 * k :] - t[: -2 * k]
    vx, vy, w = ((a[2 * k :] - a[: -2 * k]) / dt for a in (x, y, th))
    c, s = np.cos(th[k:-k]), np.sin(th[k:-k])
    return t[k:-k], np.column_stack([vx * c + vy * s, -vx * s + vy * c, w])


def commanded_at(commands: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The (forward, left, turn) last sent before each of `t` (zero before the first)."""
    i = np.searchsorted(commands[:, 0], t, side="right") - 1
    out = commands[np.clip(i, 0, None), 1:4].copy()
    out[i < 0] = 0.0
    return out


def loop_delay(commands: np.ndarray, t: np.ndarray, measured: np.ndarray) -> Optional[float]:
    """The lag in [0, MAX_LAG_S] that best correlates the command with the measured velocity:
    delay plus about half the time spent accelerating, so only a rough alignment."""
    scale = commands[:, 1:4].std(axis=0)
    if not np.any(scale > 0):
        return None
    scale[scale == 0] = 1.0
    best, best_r = None, -np.inf
    for lag in np.arange(0.0, MAX_LAG_S + 1e-9, LAG_STEP_S):
        pred = commanded_at(commands, t - lag) / scale
        meas = measured / scale
        r = np.corrcoef(pred.ravel(), meas.ravel())[0, 1]
        if np.isfinite(r) and r > best_r:
            best, best_r = float(lag), r
    return best


_DELAYS = np.arange(0.0, MAX_LAG_S + 1e-9, 0.0025)
_ACCELS = np.geomspace(0.5, 400.0, 80)  # m/s² or rad/s²
_SPEED_RANGE = np.linspace(0.96, 1.04, 17)
_FIT_S = 1.5  # at most this much of a step is fitted


def ramp(t, delay, accel, speed):
    """Displacement of the sim's robot model: still for `delay`, then `accel` up to `speed`.
    Arrays broadcast, so one call evaluates a whole grid of models."""
    tau = np.clip(t - delay, 0.0, None)
    t_ramp = np.abs(speed) / accel
    sign = np.sign(speed)
    return np.where(tau < t_ramp, 0.5 * sign * accel * tau**2, sign * accel * t_ramp**2 / 2 + speed * (tau - t_ramp))


def fit_step(t: np.ndarray, displacement: np.ndarray, speed: float) -> tuple[float, float, float]:
    """(delay, accel, speed) of `ramp` that best fit `displacement` (least squares, free offset).

    Speed is fitted too, within `_SPEED_RANGE` of `speed`: a speed off by 1 % drifts the model
    a few millimetres over a second, as much as the corner that tells delay from acceleration."""
    best, best_err = (0.0, float(_ACCELS[-1]), speed), np.inf
    delay, accel = _DELAYS[:, None, None], _ACCELS[None, :, None]
    for v in speed * _SPEED_RANGE:
        resid = displacement - ramp(t[None, None, :], delay, accel, v)
        resid -= resid.mean(axis=2, keepdims=True)
        err = (resid**2).sum(axis=2)
        i, j = np.unravel_index(int(np.argmin(err)), err.shape)
        if err[i, j] < best_err:
            best, best_err = (float(_DELAYS[i]), float(_ACCELS[j]), float(v)), err[i, j]
    return best


def _displacement(track: np.ndarray, axis: int, t_on: float) -> np.ndarray:
    """Motion since `t_on` along `axis` of the robot as it stood then (metres, or radians turned)."""
    th = np.unwrap(track[:, 4])
    before = (track[:, 0] <= t_on) & (track[:, 0] > t_on - 0.2)
    if not np.any(before):
        before = np.arange(len(track)) == 0
    x0, y0, th0 = np.median(track[before, 2]), np.median(track[before, 3]), float(np.median(th[before]))
    if axis == 2:
        return th - th0
    dx, dy = track[:, 2] - x0, track[:, 3] - y0
    c, s = math.cos(th0), math.sin(th0)
    return dx * c + dy * s if axis == 0 else -dx * s + dy * c


def steps(commands: np.ndarray, track: np.ndarray, t: np.ndarray, measured: np.ndarray, lag: float) -> list[dict]:
    """Each constant non-zero command held `MIN_STEP_S` or more, and how the robot followed it.

    The speed reached is read from the velocity (`t`, `measured`) late in the step; `lag` (a
    rough loop delay) only places that window. Delay and acceleration come from fitting `ramp`
    to the robot's positions in `track`: finite-difference velocity smears the start of a move
    by its window, which hides tens of milliseconds of delay."""
    out = []
    values = commands[:, 1:4]
    change = np.flatnonzero(np.any(np.diff(values, axis=0) != 0, axis=1)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(commands)]
    for a, b in zip(starts, ends):
        cmd = values[a]
        t_on = commands[a, 0]
        t_off = commands[b, 0] if b < len(commands) else commands[-1, 0]
        if not np.any(cmd) or t_off - t_on < MIN_STEP_S:
            continue
        axis = int(np.argmax(np.abs(cmd)))
        c = float(cmd[axis])
        settled = (t >= t_on + lag + 0.6 * (t_off - t_on)) & (t < t_off + lag)
        if not np.any(settled):
            continue
        reached = float(np.median(measured[settled, axis]))
        row = {
            "axis": AXES[axis],
            "commanded": round(c, 3),
            "reached": round(reached, 3),
            "gain": round(reached / c, 3),
            "seconds": round(float(t_off - t_on), 2),
        }
        span = (track[:, 0] >= t_on - 0.1) & (track[:, 0] < t_on + min(t_off - t_on, _FIT_S))
        if reached * c > 0 and span.sum() >= 10:
            delay, accel, speed = fit_step(track[span, 0] - t_on, _displacement(track, axis, t_on)[span], reached)
            row["delay_s"], row["accel"] = round(delay, 4), round(accel, 2)
            row["reached"], row["gain"] = round(speed, 3), round(speed / c, 3)
        out.append(row)
    return out


def _second_difference_noise(values: np.ndarray, runs: list[np.ndarray]) -> Optional[float]:
    d2 = [np.diff(values[r], 2) for r in runs if len(r) >= 10]
    d2 = np.concatenate(d2) if d2 else np.empty(0)
    return float(d2.std() / math.sqrt(6)) if len(d2) >= 30 else None


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of the consecutive True stretches of `mask`."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


def still_noise(track: np.ndarray, commands: np.ndarray) -> dict:
    """Detection noise while the robot has been sent no motion for `STILL_AFTER_S`."""
    moving = np.any(commands[:, 1:4] != 0, axis=1)
    t_moves = commands[moving, 0]
    t = track[:, 0]
    i_last = np.searchsorted(t_moves, t, side="right") - 1
    since = np.where(i_last >= 0, t - t_moves[np.clip(i_last, 0, None)], np.inf)
    runs = _runs(since >= STILL_AFTER_S)
    xy = _second_difference_noise(track[:, 2], runs), _second_difference_noise(track[:, 3], runs)
    th = _second_difference_noise(np.unwrap(track[:, 4]), runs)
    return {
        "samples": int(sum(len(r) for r in runs)),
        "position_mm": None if None in xy else round(1000 * max(xy), 2),
        "heading_deg": None if th is None else round(math.degrees(th), 3),
    }


def missing_share(track: np.ndarray, camera_frames: list[float]) -> float:
    """Share of the robot's camera's frames, between its first and last detection, without it."""
    frames = np.array(camera_frames)
    frames = frames[(frames >= track[0, 0]) & (frames <= track[-1, 0])]
    return round(1 - len(track) / len(frames), 4) if len(frames) else 0.0


def ball_noise(balls: np.ndarray) -> dict:
    """Detection noise of a ball lying still (moved < 3 mm per frame for a stretch)."""
    if len(balls) < 30:
        return {"samples": 0, "position_mm": None}
    b = _main_camera(np.column_stack([balls[:, :2], balls[:, 2:], np.zeros(len(balls))]))
    step = np.hypot(np.diff(b[:, 2]), np.diff(b[:, 3]))
    runs = [r for r in _runs(np.r_[step < 0.003, False]) if len(r) >= 30]
    xy = _second_difference_noise(b[:, 2], runs), _second_difference_noise(b[:, 3], runs)
    return {
        "samples": int(sum(len(r) for r in runs)),
        "position_mm": None if None in xy else round(1000 * max(xy), 2),
    }


def kicks(commands: np.ndarray, track: np.ndarray, balls: np.ndarray) -> list[dict]:
    """Ball top speed after each kick or chip sent with the ball at the robot."""
    kicking = commands[:, 4] != 0
    edges = np.flatnonzero(kicking & ~np.r_[False, kicking[:-1]])
    if len(balls) < 10:
        return []
    b = _main_camera(np.column_stack([balls[:, :2], balls[:, 2:], np.zeros(len(balls))]))
    k = 2
    bt = b[k:-k, 0]
    speed = np.hypot(b[2 * k :, 2] - b[: -2 * k, 2], b[2 * k :, 3] - b[: -2 * k, 3]) / (b[2 * k :, 0] - b[: -2 * k, 0])
    out = []
    for e in edges:
        t_kick = commands[e, 0]
        i_r = np.searchsorted(track[:, 0], t_kick)
        i_b = np.searchsorted(b[:, 0], t_kick)
        if i_r >= len(track) or i_b >= len(b):
            continue
        gap = math.hypot(track[i_r, 2] - b[i_b, 2], track[i_r, 3] - b[i_b, 3])
        after = (bt >= t_kick) & (bt <= t_kick + KICK_WINDOW_S) & (speed < 15.0)
        if gap > KICK_REACH_M or not np.any(after):
            continue
        out.append(
            {
                "t": round(float(t_kick), 2),
                "chip": bool(commands[e, 4] == 0x0F),
                "ball_speed": round(float(speed[after].max()), 2),
                "direction_offset_deg": kick_direction_offset(b, t_kick, float(track[i_r, 4])),
            }
        )
    return out


def kick_direction_offset(balls: np.ndarray, t_kick: float, heading: float) -> Optional[float]:
    """Degrees the ball left in anticlockwise of `heading`: from where it was at the kick to where
    it was `KICK_DIRECTION_S` later, (t, cam, x, y) rows of one camera. A robot's kicks all off the
    same way is a kicker or a tag turned on the robot, not aim."""
    i0 = np.searchsorted(balls[:, 0], t_kick)
    window = (balls[:, 0] >= t_kick + KICK_DIRECTION_S[0]) & (balls[:, 0] <= t_kick + KICK_DIRECTION_S[1])
    if i0 >= len(balls) or not np.any(window):
        return None
    end = balls[np.flatnonzero(window)[-1]]
    dx, dy = end[2] - balls[i0, 2], end[3] - balls[i0, 3]
    if math.hypot(dx, dy) < 0.05:
        return None  # the ball barely moved: no kick reached it
    off = math.atan2(dy, dx) - heading
    return round(math.degrees(math.atan2(math.sin(off), math.cos(off))), 1)


def _move_name(fwd: float, left: float, turn: float) -> str:
    if turn:
        return f"turn {turn:g} rad/s"
    if fwd or left:
        return f"drive {fwd:g}, {left:g} m/s"
    return "still"


def dribbling(commands: np.ndarray, track: np.ndarray, balls: np.ndarray, sensor: Optional[np.ndarray]) -> list[dict]:
    """Per constant command held `DRIBBLE_MIN_S` with the dribbler on: where the ball sat in the
    robot's frame, the share of ball sensor readings on, and whether it was still on at the end."""
    if commands.shape[1] < 6 or len(balls) < 10:
        return []
    b = _main_camera(np.column_stack([balls[:, :2], balls[:, 2:], np.zeros(len(balls))]))
    key = commands[:, [1, 2, 3, 5]]
    change = np.flatnonzero(np.r_[True, np.any(np.diff(key, axis=0) != 0, axis=1)])
    bounds = np.r_[change, len(commands)]
    out = []
    for a, z in zip(bounds[:-1], bounds[1:]):
        t0, t1 = commands[a, 0], commands[z - 1, 0]
        if not commands[a, 5] or t1 - t0 < DRIBBLE_MIN_S:
            continue
        in_ball = (b[:, 0] >= t0) & (b[:, 0] <= t1)
        if in_ball.sum() < 5:
            continue
        bt, bx, by = b[in_ball, 0], b[in_ball, 2], b[in_ball, 3]
        rx, ry = np.interp(bt, track[:, 0], track[:, 2]), np.interp(bt, track[:, 0], track[:, 3])
        th = np.interp(bt, track[:, 0], np.unwrap(track[:, 4]))
        dx, dy = bx - rx, by - ry
        forward, lateral = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        row = {
            "t": round(float(t0), 2),
            "move": _move_name(*commands[a, 1:4]),
            "seconds": round(float(t1 - t0), 2),
            "ball_forward_m": round(float(np.median(forward)), 3),
            "ball_left_m": round(float(np.median(lateral)), 3),
            "kept": bool(math.hypot(dx[-1], dy[-1]) < BALL_LOST_M),
        }
        if sensor is not None and len(sensor):
            on = sensor[(sensor[:, 0] >= t0) & (sensor[:, 0] <= t1), 1]
            row["sensor_on"] = round(float(on.mean()), 2) if len(on) else None
        out.append(row)
    return out


def calibrate(log: FieldLog) -> list[dict]:
    """One result per robot sent any motion, as described in the module docstring."""
    results = []
    for fw, commands in sorted(log.commands.items()):
        if not np.any(commands[:, 1:4] != 0):
            continue
        robot = log.robot_of(fw)
        if robot is None or robot not in log.robots:
            results.append({"firmware_id": fw, "error": "not seen by vision (no run info or never detected)"})
            continue
        track = _main_camera(log.robots[robot])
        t, measured = local_velocity(track)
        rough = loop_delay(commands, t, measured)
        step_rows = steps(commands, track, t, measured, rough or 0.0)
        delays = [r["delay_s"] for r in step_rows if "delay_s" in r]
        lag = round(float(np.median(delays)), 3) if delays else None
        summary = {}
        for axis in AXES:
            rows = [r for r in step_rows if r["axis"] == axis]
            accel = [r["accel"] for r in rows if "accel" in r]
            if rows:
                summary[axis] = {
                    "gain": round(float(np.median([r["gain"] for r in rows])), 3),
                    "accel": round(float(np.median(accel)), 2) if accel else None,
                    "steps": len(rows),
                }
        robot_kicks = kicks(commands, track, log.balls)
        offsets = [k["direction_offset_deg"] for k in robot_kicks if k["direction_offset_deg"] is not None]
        results.append(
            {
                "robot": f"{robot[0]} {robot[1]}",
                "firmware_id": fw,
                "loop_delay_s": lag,
                "summary": summary,
                "vision": {
                    **still_noise(track, commands),
                    "missing_share": missing_share(track, log.camera_frames[int(track[0, 1])]),
                },
                "steps": step_rows,
                "kicks": robot_kicks,
                "kick_direction_offset_median_deg": round(float(np.median(offsets)), 1) if offsets else None,
                "dribbling": dribbling(commands, track, log.balls, log.sensor.get(fw)),
                "kick_speed_median": (
                    round(float(np.median([k["ball_speed"] for k in robot_kicks if not k["chip"]])), 2)
                    if any(not k["chip"] for k in robot_kicks)
                    else None
                ),
            }
        )
    return results


def report(results: list[dict], ball: dict) -> str:
    lines = []
    for r in results:
        if "error" in r:
            lines.append(f"firmware {r['firmware_id']}: {r['error']}")
            continue
        lines.append(f"{r['robot']} (firmware {r['firmware_id']})")
        lag = r["loop_delay_s"]
        lines.append(f"  loop delay        {'-' if lag is None else f'{lag * 1000:.0f} ms'}")
        for axis, s in r["summary"].items():
            unit = "rad/s²" if axis == "turn" else "m/s²"
            accel = "-" if s["accel"] is None else f"{s['accel']} {unit}"
            lines.append(f"  {axis:<8} gain {s['gain']:<6} accel {accel:<14} ({s['steps']} steps)")
        v = r["vision"]
        lines.append(
            f"  vision at rest    {v['position_mm']} mm, {v['heading_deg']}°, "
            f"{v['missing_share']:.1%} frames missing ({v['samples']} samples)"
        )
        if r["kicks"]:
            lines.append(
                f"  kicks             {len(r['kicks'])}, median ball speed {r['kick_speed_median']} m/s, "
                f"direction {r['kick_direction_offset_median_deg']}° off the heading (+ anticlockwise)"
            )
            for k in r["kicks"]:
                lines.append(f"    t {k['t']:>7} s  {k['ball_speed']} m/s  {k['direction_offset_deg']}°")
        for d in r["dribbling"]:
            lines.append(
                f"  dribble {d['move']:<20} ball {d['ball_forward_m']} m ahead, {d['ball_left_m']} m left, "
                f"sensor on {d.get('sensor_on', '-')}, {'kept' if d['kept'] else 'LOST'} ({d['seconds']} s)"
            )
        for s in r["steps"]:
            lines.append(
                f"    {s['axis']:<8} {s['commanded']:>6} -> {s['reached']:>6} "
                f"(gain {s['gain']}, accel {s.get('accel', '-')}, delay {s.get('delay_s', '-')} s, {s['seconds']} s)"
            )
    lines.append(f"ball at rest        {ball['position_mm']} mm ({ball['samples']} samples)")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, help="default: <log>.calibration.json beside the log")
    args = parser.parse_args(argv)
    log = FieldLog(args.log)
    results = calibrate(log)
    ball = ball_noise(log.balls)
    print(report(results, ball))
    out = args.json or args.log.with_suffix(".calibration.json")
    out.write_text(json.dumps({"log": str(args.log), "run_info": log.info, "robots": results, "ball": ball}, indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    main()
