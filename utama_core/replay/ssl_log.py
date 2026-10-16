"""Official RoboCup SSL game logs (`*.log.gz`) as replays in our columnar format.

The league publishes every official match as a game log: the network packets seen during the
match, gzipped. `convert_log` turns one into the `<stem>.npz` + `<stem>.sparse_referee.pkl` pair
a round-robin match writes, through `ColumnarReplayWriter`, so `load_columnar_replay`,
`render_window`, `scenario_from_replay` and the analyses read a real match like one of ours.
`tools/convert_ssl_log.py` converts files or downloads the published logs.

Log format: the header `SSL_LOG_FILE` and a big-endian int32 version, then records of int64
receive time [ns], int32 message type, int32 size and the serialized message. Types read here:
3 referee (`Referee`), 4 vision (`SSL_WrapperPacket`; type 2 is the same detection in the
pre-2014 wrapper, whose geometry layout differs) and 5 tracker (`TrackerWrapperPacket`).

What a converted replay holds:
- Perspective: yellow is the friendly team (`my_team_is_yellow`), blue the enemy: the teams
  as they were coloured at kick-off. Some matches swap colours at half time (both teams change
  markers; the game controller moves names, scores and cards with them); from a swap on, the
  converter swaps robots, team info, sides and coloured commands back (`_colours_swapped`), so
  friendly stays one team and its score never falls. Only `game_events` keep the colours they
  were sent with.
  `side_is_right[t]` (yellow defends the +x goal) is the referee's `blue_team_on_positive_half`
  negated; for a message without that field (logs before 2019) it is where yellow's goalkeeper
  stood during that stage. The game controller's sides are set by hand, and one 2022 match ran
  with them the wrong way round throughout: where yellow's goalkeeper and yellow's robots as a
  whole both played on the other half (`_contradicted_sides`), the converter inverts the sides,
  in `side_is_right` and the referee's `blue_team_on_positive_half` alike.
- Time: 60 Hz ticks on the log's receive clock, `ts` 0 at the first referee message. The
  window ends where play ends: a penalty shoot-out and the post-game stage are dropped. Breaks,
  timeouts and halts stay in, so `ts` has no gaps; skip them by `stage`/`referee_command`.
- Robots and ball: from the tracker, an autoref's filtered world in metres with velocities, the
  source with the most frames when several autorefs ran. A tick takes the newest tracked frame
  at or before it; after `_STALE_S` without one, robots are absent. Logs without a tracker
  (before 2020) merge the cameras' raw detections instead and difference positions over
  `_DIFF_TICKS` ticks for velocities. Accelerations are differenced velocities in both cases.
  The ball holds its last position (at rest) while no ball is seen, or the one seen is beyond the
  field's walls: a spare ball, or a tracker's chip estimate gone wild (some 2022 frames put the
  ball tens of metres up and away for a few ticks).
- Robot ids: one column per id a team used in the match (`friendly_ids`/`enemy_ids`, any of
  0-15), NaN while that robot is off the field, as in our own replays. A substitution is one id
  going NaN and another appearing, so no robot is lost; there is no fixed 6-slot layout, and
  more than 6 columns is normal.
- `has_ball`: real logs carry no dribbler sensor, so both teams use the vision-proximity rule of
  `RobotInfoRefiner` for robots without a trusted IR sensor (robot centre within
  `_BALL_CAPTURE_DIST` of the ball). It is looser than rsim's contact flag.
- Referee: a message wherever command, stage, score, goalkeeper or sides change (the game
  controller resends its state about 66 times a second), with the fields
  `RefereeMessageReceiver` fills, `stage_time_left` in seconds, `designated_position` in metres
  and `game_events` as dicts (`MessageToDict`: the generated classes cannot be pickled).
  Commands keep their protocol values, including `GOAL_*` and, before 2021, `INDIRECT_FREE_*`,
  with one change: a free kick reads `NORMAL_START` once it is taken (`free_kick_taken`), as in
  our referee. The real game controller keeps the free-kick command through the play after it.
- `<stem>.source.json`: the log, teams, final score, tracker source and field geometry, and
  under `extras` which of the sidecars below were written.

Sidecars, for measuring physics and vision rather than replaying (`tools/convert_ssl_log.py`
can add them to a replay already converted). Teams are in kick-off colours like the replay
(yellow friendly), positions in metres in field coordinates, times `ts` in seconds on the
replay's clock. The arrays are `np.savez_compressed` archives named `.zip`, so a glob for
replays (`*.npz`) skips them; `np.load` reads them as usual.
- `<stem>.kicks.json` (tracker logs): every kick the tracker reported (`KickedBall`), once per
  kick: start time, kicker (the last robot the tracker named; `kickers` counts how many it
  named), start position, the first and the last (most refined) velocity estimate, on how
  many ticks it was reported, and the stop time and position when the tracker gave them.
- `<stem>.tracker.zip` (tracker logs): per tick, the `visibility` (0-1) of the ball and every
  robot, and robots' angular velocity, NaN where not given. Robot arrays are `[tick, team, id]`,
  team 0 yellow and 1 blue, ids 0-15.
Not every autoref's tracker sends kicks or visibility (in 2026 ER-Force sent neither, TIGERs
both), so each comes from the replay's tracker if it sends it, else from the source that sends
it most (`source.json`'s `extras.kicks_source` / `visibility_source`); angular velocity always
from the replay's tracker.
- `<stem>.vision.zip` (`raw_vision=True`): every camera detection inside the replay window,
  unfiltered: one row per camera frame (`frame_*`: receive time `ts`, capture and send time on
  the camera clock, camera id, frame number), per ball detection (`ball_*`) and per robot
  detection (`robot_*`), with confidence, pixel position and the row of the frame they came in.
"""

from __future__ import annotations

import array
import dataclasses
import gzip
import json
import math
import struct
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from google.protobuf.json_format import MessageToDict

from utama_core.data_processing.refiners.robot_info import _BALL_CAPTURE_DIST
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.columnar_reader import advance_clocks
from utama_core.replay.columnar_writer import (
    ColumnarReplayWriter,
    ColumnarReplayWriterConfig,
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

LOG_HEADER = b"SSL_LOG_FILE"
VISION_2010, REFEREE, VISION_2014, TRACKER = 2, 3, 4, 5

TICK_S = 1 / 60  # our replays' rate
_STALE_S = 0.5  # no tracked frame for this long: robots are absent
_CAMERA_STALE_S = 0.1  # a camera's detection older than this is not merged
_BALL_GATE_M = 1.0  # vision fallback: how far from the last tick's ball a detection is the same ball
_DIFF_TICKS = 3  # velocity and acceleration: central difference over +-3 ticks (0.1 s)
_MAX_ID = 16  # robot ids 0-15 (the standard pattern set)
_SIDE_CHECK_S = 10.0  # live play needed to judge the referee's sides by where the robots played
_SWAP_MIN_S = 60.0  # team names swapped for less than this are an operator's slip, not a colour swap
_DIV_B_WALLS = (4.8, 3.3)  # |x|, |y| of the walls of a 9 x 6 m field, for a log without geometry
_YELLOW, _BLUE = 0, 1  # team index in the arrays below; the tracker's `Team` is 1 yellow, 2 blue
_END_STAGES = frozenset({Stage.PENALTY_SHOOTOUT_BREAK.value, Stage.PENALTY_SHOOTOUT.value, Stage.POST_GAME.value})
FREE_KICKS = frozenset(
    c.value
    for c in (
        RefereeCommand.DIRECT_FREE_YELLOW,
        RefereeCommand.DIRECT_FREE_BLUE,
        RefereeCommand.INDIRECT_FREE_YELLOW,
        RefereeCommand.INDIRECT_FREE_BLUE,
    )
)
IN_PLAY_M = 0.05  # the rulebook's ball in play: moved at least 0.05 m
_SAME_KICK_S = 0.05  # tracker kick reports whose start times differ by less than this are one kick


def read_records(path) -> Iterator[tuple[int, int, bytes]]:
    """(receive time [ns], message type, payload) of every record in a log, gzipped or not.
    A record cut off at the end of the file (a logger stopped mid-write) is dropped."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(len(LOG_HEADER) + 4)
        if header[: len(LOG_HEADER)] != LOG_HEADER:
            raise ValueError(f"{path} is not an SSL game log (header {header[:len(LOG_HEADER)]!r})")
        while True:
            head = f.read(16)
            if len(head) < 16:
                return
            ts_ns, msg_type, size = struct.unpack(">qii", head)
            payload = f.read(size)
            if len(payload) < size:
                return
            yield ts_ns, msg_type, payload


def log_stem(name: str) -> str:
    """A log file's name without `.log.gz`, percent-escapes decoded (some published names
    carry them, e.g. `Rob%C3%B4Cin`)."""
    stem = name.removesuffix(".gz").removesuffix(".log")
    while (decoded := urllib.parse.unquote(stem)) != stem:
        stem = decoded
    return stem


def _referee_key(msg: Referee) -> tuple:
    """What makes a referee message worth keeping: everything but its clocks."""
    return (
        msg.command_counter,
        msg.command,
        msg.stage,
        msg.yellow.score,
        msg.blue.score,
        msg.yellow.goalkeeper,
        msg.blue.goalkeeper,
        msg.blue_team_on_positive_half if msg.HasField("blue_team_on_positive_half") else None,
    )


def _team_info(name: str, packet) -> TeamInfo:
    info = TeamInfo(name)
    info.parse_referee_packet(packet)
    return info


def _recoloured(command: int, swapped: bool) -> RefereeCommand:
    """A command id as a `RefereeCommand`, its team colour swapped if `swapped` (the coloured
    commands come in yellow/blue pairs from 4 to 17)."""
    if swapped and command >= RefereeCommand.PREPARE_KICKOFF_YELLOW.value:
        command ^= 1
    return RefereeCommand.from_id(command)


def referee_data(msg: Referee, received_s: float, swapped: bool = False) -> RefereeData:
    """`msg` as `RefereeData`, field for field as `RefereeMessageReceiver._update_data`, but with
    the protocol's units converted: `stage_time_left` us -> s, `designated_position` mm -> m.
    `swapped`: the teams wear each other's colours, so swap them back (`_colours_swapped`)."""
    yellow, blue = (msg.blue, msg.yellow) if swapped else (msg.yellow, msg.blue)
    blue_on_positive = msg.blue_team_on_positive_half if msg.HasField("blue_team_on_positive_half") else None
    return RefereeData(
        source_identifier=msg.source_identifier,
        time_sent=msg.packet_timestamp / 1e6,
        time_received=received_s,
        referee_command=_recoloured(msg.command, swapped),
        referee_command_timestamp=msg.command_timestamp / 1e6,
        stage=Stage.from_id(msg.stage),
        stage_time_left=msg.stage_time_left / 1e6,
        blue_team=_team_info("blue", blue),
        yellow_team=_team_info("yellow", yellow),
        designated_position=(
            (msg.designated_position.x / 1000, msg.designated_position.y / 1000)
            if msg.HasField("designated_position")
            else None
        ),
        blue_team_on_positive_half=None if blue_on_positive is None else blue_on_positive != swapped,
        next_command=_recoloured(msg.next_command, swapped) if msg.HasField("next_command") else None,
        current_action_time_remaining=(
            msg.current_action_time_remaining if msg.HasField("current_action_time_remaining") else None
        ),
        game_events=[MessageToDict(e, preserving_proto_field_name=True) for e in msg.game_events],
        match_type=msg.match_type,
        status_message=msg.status_message or None,
    )


class _Scan:
    """One pass over a log: the referee messages that change something, every tracker source's
    frames, the camera detections (kept only without a tracker) and the field geometry; with
    `raw_vision`, every camera detection as `_VisionRows` too."""

    def __init__(self, path: Path, raw_vision: bool = False):
        self.referee: list[tuple[int, Referee]] = []
        self.trackers: dict[str, list[tuple[int, bytes]]] = defaultdict(list)
        self.vision: dict[int, list[tuple[int, bytes]]] = defaultdict(list)  # by record type
        self.raw_vision = _VisionRows() if raw_vision else None
        self.kick_frames: Counter[str] = Counter()  # per tracker source, frames with a `KickedBall`
        self.visibility_frames: Counter[str] = Counter()  # ... and with the ball's visibility
        self.geometry = None
        self.last_ns = 0
        last_key = None
        for ts_ns, msg_type, payload in read_records(path):
            if msg_type == REFEREE:
                msg = Referee()
                msg.ParseFromString(payload)
                key = _referee_key(msg)
                if key != last_key:
                    self.referee.append((ts_ns, msg))
                    last_key = key
            elif msg_type == TRACKER:
                packet = TrackerWrapperPacket()
                packet.ParseFromString(payload)
                if packet.HasField("tracked_frame"):
                    name, frame = packet.source_name or packet.uuid, packet.tracked_frame
                    self.trackers[name].append((ts_ns, payload))
                    self.vision.clear()
                    # not every autoref's tracker sends these (2026: ER-Force neither, TIGERs both)
                    self.kick_frames[name] += frame.HasField("kicked_ball")
                    self.visibility_frames[name] += bool(frame.balls) and frame.balls[0].HasField("visibility")
            elif msg_type in (VISION_2010, VISION_2014):
                if not self.trackers:  # camera detections are only read when there is no tracker
                    self.vision[msg_type].append((ts_ns, payload))
                if msg_type == VISION_2014 and (self.geometry is None or self.raw_vision is not None):
                    packet = SSL_WrapperPacket()
                    packet.ParseFromString(payload)
                    if packet.HasField("geometry") and self.geometry is None:
                        self.geometry = packet.geometry.field
                    if packet.HasField("detection") and self.raw_vision is not None:
                        self.raw_vision.add(ts_ns, packet.detection)
            else:
                continue  # the index record (type 6) has no meaningful time
            self.last_ns = max(self.last_ns, ts_ns)


class _VisionRows:
    """Camera detections as flat typed columns, appended while scanning: a match's vision packets
    kept as messages would not fit in memory. Positions in millimetres, as sent."""

    FRAME = {"recv_ns": "q", "t_capture": "d", "t_sent": "d", "camera": "i", "number": "q"}
    BALL = {"frame": "q", "x": "f", "y": "f", "z": "f", "pixel_x": "f", "pixel_y": "f", "confidence": "f", "area": "i"}
    ROBOT = {
        "frame": "q",
        "team": "b",
        "id": "i",
        "x": "f",
        "y": "f",
        "orientation": "f",
        "pixel_x": "f",
        "pixel_y": "f",
        "height": "f",
        "confidence": "f",
    }

    def __init__(self):
        self.columns = {
            f"{table}_{name}": array.array(code)
            for table, fields in (("frame", self.FRAME), ("ball", self.BALL), ("robot", self.ROBOT))
            for name, code in fields.items()
        }

    def add(self, recv_ns: int, detection) -> None:
        c = self.columns
        row = len(c["frame_recv_ns"])
        for name, value in (
            ("recv_ns", recv_ns),
            ("t_capture", detection.t_capture),
            ("t_sent", detection.t_sent),
            ("camera", detection.camera_id),
            ("number", detection.frame_number),
        ):
            c[f"frame_{name}"].append(value)
        for b in detection.balls:
            for name, value in (
                ("frame", row),
                ("x", b.x),
                ("y", b.y),
                ("z", b.z),
                ("pixel_x", b.pixel_x),
                ("pixel_y", b.pixel_y),
                ("confidence", b.confidence),
                ("area", b.area if b.HasField("area") else -1),
            ):
                c[f"ball_{name}"].append(value)
        for team, robots in ((_YELLOW, detection.robots_yellow), (_BLUE, detection.robots_blue)):
            for r in robots:
                for name, value in (
                    ("frame", row),
                    ("team", team),
                    ("id", r.robot_id if r.HasField("robot_id") else -1),
                    ("x", r.x),
                    ("y", r.y),
                    ("orientation", r.orientation if r.HasField("orientation") else math.nan),
                    ("pixel_x", r.pixel_x),
                    ("pixel_y", r.pixel_y),
                    ("height", r.height if r.HasField("height") else math.nan),
                    ("confidence", r.confidence),
                ):
                    c[f"robot_{name}"].append(value)

    def arrays(self, start_ns: int, end_ns: int, tick_ns: np.ndarray, swapped: np.ndarray) -> dict[str, np.ndarray]:
        """The frames received inside [start_ns, end_ns) and their detections: receive times as
        `frame_ts` on the replay clock, positions in metres, teams in kick-off colours."""
        a = {name: np.frombuffer(col, dtype=col.typecode) for name, col in self.columns.items()}
        recv = a["frame_recv_ns"]
        keep = (recv >= start_ns) & (recv < end_ns)
        new_row = np.cumsum(keep) - 1  # a kept frame's row among the kept frames
        tick = np.clip(np.searchsorted(tick_ns, recv, side="right") - 1, 0, max(len(tick_ns) - 1, 0))
        frame_swapped = swapped[tick] if len(swapped) else np.zeros(len(recv), dtype=bool)
        out = {"frame_ts": (recv[keep] - start_ns) / 1e9}
        out.update({f"frame_{name}": a[f"frame_{name}"][keep] for name in ("t_capture", "t_sent", "camera", "number")})
        for table, fields in (("ball", self.BALL), ("robot", self.ROBOT)):
            rows = a[f"{table}_frame"]
            kept = keep[rows]
            for name in fields:
                column = a[f"{table}_{name}"][kept]
                out[f"{table}_{name}"] = column / 1000 if name in ("x", "y", "z") else column
            out[f"{table}_frame"] = new_row[rows[kept]]
            if table == "robot":
                out["robot_team"] = (out["robot_team"] ^ frame_swapped[rows[kept]]).astype(np.int8)
        return out


class _World:
    """Robots and ball on the tick grid. Robot arrays are [tick, team, id, ...], NaN where absent."""

    def __init__(self, n_ticks: int):
        self.robot_p = np.full((n_ticks, 2, _MAX_ID, 2), np.nan)
        self.robot_v = np.full((n_ticks, 2, _MAX_ID, 2), np.nan)
        self.robot_orientation = np.full((n_ticks, 2, _MAX_ID), np.nan)
        self.ball_p = np.full((n_ticks, 3), np.nan)
        self.ball_v = np.full((n_ticks, 3), np.nan)
        self.ball_seen = np.zeros(n_ticks, dtype=bool)
        # from a tracker only (NaN from camera detections): the `.tracker.zip` and `.kicks.json` sidecars
        self.ball_visibility = np.full(n_ticks, np.nan)
        self.robot_visibility = np.full((n_ticks, 2, _MAX_ID), np.nan)
        self.robot_vel_angular = np.full((n_ticks, 2, _MAX_ID), np.nan)
        self.kicks: list[dict] = []  # see `_add_kick`

    def finish(self) -> None:
        """Fill what the source did not give: missing velocities from positions, accelerations
        from velocities, the ball's position while unseen, and `has_ball`."""
        missing = np.isnan(self.robot_v) & ~np.isnan(self.robot_p)
        self.robot_v[missing] = _central_difference(self.robot_p)[missing]
        seen = np.flatnonzero(self.ball_seen)
        if len(seen):
            # held at its last position, and at its first before it was ever seen
            held = seen[np.maximum(np.searchsorted(seen, np.arange(len(self.ball_p)), side="right") - 1, 0)]
            unseen = ~self.ball_seen
            self.ball_p[unseen] = self.ball_p[held[unseen]]
            self.ball_v[unseen] = 0.0
        missing = np.isnan(self.ball_v) & ~np.isnan(self.ball_p)
        self.ball_v[missing] = _central_difference(self.ball_p)[missing]
        for v, p in ((self.robot_v, self.robot_p), (self.ball_v, self.ball_p)):
            v[np.isnan(v) & ~np.isnan(p)] = 0.0  # the ends of a presence, where no difference exists
        self.robot_a = _central_difference(self.robot_v)
        self.ball_a = _central_difference(self.ball_v)
        for a, p in ((self.robot_a, self.robot_p), (self.ball_a, self.ball_p)):
            a[np.isnan(a) & ~np.isnan(p)] = 0.0
        gap = self.robot_p - self.ball_p[:, None, None, :2]
        self.has_ball = np.hypot(gap[..., 0], gap[..., 1]) < _BALL_CAPTURE_DIST


def _central_difference(x: np.ndarray, k: int = _DIFF_TICKS) -> np.ndarray:
    """d/dt along axis 0 over +-k ticks; NaN where either end is missing."""
    out = np.full_like(x, np.nan)
    out[k:-k] = (x[2 * k :] - x[: -2 * k]) / (2 * k * TICK_S)
    return out


def _latest(times_ns: np.ndarray, tick_ns: np.ndarray, stale_s: float) -> np.ndarray:
    """Per tick, the index of the newest time at or before it, or -1 if none within `stale_s`."""
    i = np.searchsorted(times_ns, tick_ns, side="right") - 1
    ok = i >= 0
    ok[ok] = (tick_ns[ok] - times_ns[i[ok]]) <= stale_s * 1e9
    return np.where(ok, i, -1)


def _tracker_world(frames: list[tuple[int, bytes]], tick_ns: np.ndarray) -> tuple[_World, int]:
    """The world from one tracker source's frames; also the count of robots with an id past 15."""
    world = _World(len(tick_ns))
    times = np.array([t for t, _ in frames], dtype=np.int64)
    latest = _latest(times, tick_ns, _STALE_S)
    dropped = 0
    packet = TrackerWrapperPacket()
    decoded = -1
    for k, i in enumerate(latest):
        if i < 0:
            continue
        new_frame = i != decoded
        if new_frame:
            packet.ParseFromString(frames[i][1])
            decoded = i
        frame = packet.tracked_frame
        for robot in frame.robots:
            team, rid = robot.robot_id.team - 1, robot.robot_id.id
            if team not in (_YELLOW, _BLUE) or rid >= _MAX_ID:
                dropped += 1
                continue
            world.robot_p[k, team, rid] = (robot.pos.x, robot.pos.y)
            world.robot_orientation[k, team, rid] = robot.orientation
            if robot.HasField("vel"):
                world.robot_v[k, team, rid] = (robot.vel.x, robot.vel.y)
            if robot.HasField("vel_angular"):
                world.robot_vel_angular[k, team, rid] = robot.vel_angular
            if robot.HasField("visibility"):
                world.robot_visibility[k, team, rid] = robot.visibility
        if frame.balls:
            ball = frame.balls[0]  # the primary ball
            world.ball_p[k] = (ball.pos.x, ball.pos.y, ball.pos.z)
            world.ball_seen[k] = True
            if ball.HasField("vel"):
                world.ball_v[k] = (ball.vel.x, ball.vel.y, ball.vel.z)
            if ball.HasField("visibility"):
                world.ball_visibility[k] = ball.visibility
        if new_frame and frame.HasField("kicked_ball"):  # a kick lasts many frames: those between ticks can go
            _add_kick(world.kicks, frame.kicked_ball, frames[i][0], frame.timestamp)
    return world, dropped


def _add_kick(kicks: list[dict], kick, recv_ns: int, frame_timestamp: float) -> None:
    """Fold one tracked frame's `KickedBall` into `kicks`. The tracker repeats a kick in every
    frame until the ball stops, refining it: a report starting within `_SAME_KICK_S` of the last
    kick's start is that kick, with its velocity, stop and kicker updated (the tracker sometimes
    gives a kick to another robot a few frames in; `kickers` counts the robots it named).
    Times move to the log's receive clock by the frame's receive time minus its tracker time."""
    kicker = (kick.robot_id.team - 1, kick.robot_id.id) if kick.HasField("robot_id") else None
    last = kicks[-1] if kicks else None
    if last is None or abs(kick.start_timestamp - last["start_timestamp"]) >= _SAME_KICK_S:
        last = {
            "start_timestamp": kick.start_timestamp,
            "start_ns": recv_ns - round((frame_timestamp - kick.start_timestamp) * 1e9),
            "kicker": kicker,
            "pos": [kick.pos.x, kick.pos.y],
            "vel_first": [kick.vel.x, kick.vel.y, kick.vel.z],
            "reports": 0,
            "kickers": set(),
        }
        kicks.append(last)
    last["kicker"] = kicker
    last["kickers"].add(kicker)
    last["vel"] = [kick.vel.x, kick.vel.y, kick.vel.z]
    last["reports"] += 1
    if kick.HasField("stop_timestamp"):
        last["stop_ns"] = recv_ns - round((frame_timestamp - kick.stop_timestamp) * 1e9)
    if kick.HasField("stop_pos"):
        last["stop_pos"] = [kick.stop_pos.x, kick.stop_pos.y]


def _vision_world(packets: list[tuple[int, bytes]], tick_ns: np.ndarray) -> tuple[_World, int]:
    """The world merged from raw camera detections: per tick each camera's newest frame (within
    `_CAMERA_STALE_S`), a robot seen by several cameras at the confidence-weighted mean, the ball
    as the detection nearest the last tick's ball (within `_BALL_GATE_M`), else the most confident
    one. No filtering beyond that."""
    by_camera: dict[int, list[tuple[int, object]]] = defaultdict(list)
    wrapper = SSL_WrapperPacket()
    for ts_ns, payload in packets:
        wrapper.ParseFromString(payload)
        if wrapper.HasField("detection"):
            detection = wrapper.detection
            robots = [
                (team, r.robot_id, r.x / 1000, r.y / 1000, r.orientation, r.confidence)
                for team, robots in ((_YELLOW, detection.robots_yellow), (_BLUE, detection.robots_blue))
                for r in robots
                if r.HasField("robot_id")
            ]
            balls = [(b.x / 1000, b.y / 1000, b.z / 1000, b.confidence) for b in detection.balls]
            by_camera[detection.camera_id].append((ts_ns, (robots, balls)))
    world = _World(len(tick_ns))
    latest = {
        cam: (_latest(np.array([t for t, _ in rows], dtype=np.int64), tick_ns, _CAMERA_STALE_S), rows)
        for cam, rows in by_camera.items()
    }
    dropped, previous_ball = 0, None
    for k in range(len(tick_ns)):
        sums: dict[tuple[int, int], list[float]] = {}
        balls_seen = []
        for index, rows in latest.values():
            if index[k] < 0:
                continue
            robots, balls = rows[index[k]][1]
            for team, rid, x, y, orientation, confidence in robots:
                if rid >= _MAX_ID:
                    dropped += 1
                    continue
                w = max(confidence, 1e-3)
                s = sums.setdefault((team, rid), [0.0, 0.0, 0.0, 0.0, 0.0])
                s[0] += w * x
                s[1] += w * y
                s[2] += w * math.cos(orientation)
                s[3] += w * math.sin(orientation)
                s[4] += w
            balls_seen += balls
        for (team, rid), (sx, sy, sc, ss, w) in sums.items():
            world.robot_p[k, team, rid] = (sx / w, sy / w)
            world.robot_orientation[k, team, rid] = math.atan2(ss, sc)
        if balls_seen:
            ball = max(balls_seen, key=lambda b: b[3])
            if previous_ball is not None:
                nearest = min(balls_seen, key=lambda b: math.hypot(b[0] - previous_ball[0], b[1] - previous_ball[1]))
                if math.hypot(nearest[0] - previous_ball[0], nearest[1] - previous_ball[1]) < _BALL_GATE_M:
                    ball = nearest
            world.ball_p[k] = previous_ball = ball[:3]
            world.ball_seen[k] = True
    return world, dropped


def _colours_swapped(referee: list[tuple[int, Referee]]) -> np.ndarray:
    """Per referee message, whether the teams wear each other's colours compared with kick-off:
    its team names are those of the first message naming two teams, swapped, and stay so for at
    least `_SWAP_MIN_S` (a 2018 operator swapped them for 3 s and back, robots unchanged)."""
    names = [(msg.yellow.name, msg.blue.name) for _, msg in referee]
    first = next(((yellow, blue) for yellow, blue in names if yellow and blue and yellow != blue), None)
    swapped = np.array([first is not None and pair == first[::-1] for pair in names], dtype=bool)
    times = np.array([t for t, _ in referee], dtype=np.int64)
    for k0 in np.flatnonzero(swapped & ~np.r_[False, swapped[:-1]]):
        k1 = k0 + 1
        while k1 < len(swapped) and swapped[k1]:
            k1 += 1
        if times[min(k1, len(times) - 1)] - times[k0] < _SWAP_MIN_S * 1e9:
            swapped[k0:k1] = False
    return swapped


def _robot_sides(world: _World, ticks: np.ndarray, keepers: np.ndarray) -> tuple[Optional[bool], Optional[bool]]:
    """Whether yellow defends +x judging by where its robots stood at `ticks`: by the median x of
    yellow's goalkeeper, and by the mean x of yellow's robots against blue's (None: not seen)."""
    keeper_x = world.robot_p[ticks, _YELLOW, np.minimum(keepers[ticks], _MAX_ID - 1), 0]
    keeper_x = keeper_x[~np.isnan(keeper_x)]
    yellow = world.robot_p[ticks, _YELLOW, :, 0]
    blue = world.robot_p[ticks, _BLUE, :, 0]
    keeper = bool(np.median(keeper_x) > 0) if len(keeper_x) else None
    mass = (
        bool(np.nanmean(yellow) > np.nanmean(blue)) if np.isfinite(yellow).any() and np.isfinite(blue).any() else None
    )
    return keeper, mass


def _sides_from_goalkeepers(world: _World, stage: np.ndarray, keepers: np.ndarray) -> np.ndarray:
    """Per tick whether yellow defends +x, for logs whose referee does not say: within each
    stage, the side of the median x of yellow's goalkeeper (else of yellow's robots relative to
    blue's). A stage with no robots seen keeps the previous stage's answer."""
    sides = np.ones(len(stage), dtype=bool)
    previous = True
    for value in np.unique(stage):
        ticks = np.flatnonzero(stage == value)
        keeper, mass = _robot_sides(world, ticks, keepers)
        previous = next((side for side in (keeper, mass) if side is not None), previous)
        sides[ticks] = previous
    return sides


def _contradicted_sides(world: _World, side_is_right: np.ndarray, live: np.ndarray, keepers: np.ndarray) -> np.ndarray:
    """Per tick, whether the referee's sides are the wrong way round: within each run of ticks with
    the same `side_is_right`, yellow's goalkeeper and yellow's robots as a whole both stood on the
    other half through the run's live play, of which there is at least `_SIDE_CHECK_S`."""
    wrong = np.zeros(len(side_is_right), dtype=bool)
    starts = np.flatnonzero(np.r_[True, side_is_right[1:] != side_is_right[:-1]])
    for k0, k1 in zip(starts, np.r_[starts[1:], len(side_is_right)]):
        ticks = k0 + np.flatnonzero(live[k0:k1])
        if len(ticks) < round(_SIDE_CHECK_S / TICK_S):
            continue
        keeper, mass = _robot_sides(world, ticks, keepers)
        wrong[k0:k1] = keeper is not None and keeper == mass != side_is_right[k0]
    return wrong


def free_kick_taken(command: np.ndarray, counter: np.ndarray, ball_xy: np.ndarray) -> np.ndarray:
    """Per tick, whether the free kick in progress has been taken: from the first tick its ball
    has moved `IN_PLAY_M` from where it stood when the free kick was given, to the end of that
    command (`counter`, the referee's command counter, separates two free kicks in a row).

    The game controller keeps `DIRECT_FREE_*` through the play that follows, until the next
    stoppage. Our referee switches to `NORMAL_START` before the kick instead (its advance 3, see
    `docs/custom_referee.md`), and the analyses count live play as `NORMAL_START`/`FORCE_START`,
    so a converted replay says `NORMAL_START` from the kick on."""
    taken = np.zeros(len(command), dtype=bool)
    in_kick = np.isin(command, list(FREE_KICKS))
    starts = np.flatnonzero(in_kick & np.r_[True, (command[1:] != command[:-1]) | (counter[1:] != counter[:-1])])
    for k0 in starts:
        k1 = k0 + 1
        while k1 < len(command) and command[k1] == command[k0] and counter[k1] == counter[k0]:
            k1 += 1
        gap = ball_xy[k0:k1] - ball_xy[k0]
        moved = np.hypot(gap[:, 0], gap[:, 1]) >= IN_PLAY_M
        if moved.any():
            taken[k0 + int(moved.argmax()) : k1] = True
    return taken


def _robots(world: _World, k: int, team: int, is_friendly: bool) -> dict[int, Robot]:
    robots = {}
    for rid in np.flatnonzero(~np.isnan(world.robot_p[k, team, :, 0])):
        rid = int(rid)
        robots[rid] = Robot(
            id=rid,
            is_friendly=is_friendly,
            has_ball=bool(world.has_ball[k, team, rid]),
            p=Vector2D(*world.robot_p[k, team, rid]),
            v=Vector2D(*world.robot_v[k, team, rid]),
            a=Vector2D(*world.robot_a[k, team, rid]),
            orientation=float(world.robot_orientation[k, team, rid]),
        )
    return robots


def _geometry(field) -> Optional[dict]:
    if field is None:
        return None
    geometry = {
        "field_length": field.field_length / 1000,
        "field_width": field.field_width / 1000,
        "goal_width": field.goal_width / 1000,
        "goal_depth": field.goal_depth / 1000,
        "boundary_width": field.boundary_width / 1000,
    }
    if field.HasField("penalty_area_depth"):
        geometry["penalty_area_depth"] = field.penalty_area_depth / 1000
        geometry["penalty_area_width"] = field.penalty_area_width / 1000
    return geometry


def convert_log(
    log_path, out_dir, stem: Optional[str] = None, raw_vision: bool = False, write_replay: bool = True
) -> dict:
    """Convert one game log to `<out_dir>/<stem>.npz` (+ `.sparse_referee.pkl`, `.source.json`)
    and the sidecars (`.kicks.json`, `.tracker.zip`, and `.vision.zip` with `raw_vision`).
    Returns what `source.json` holds. See the module docstring for what each file contains.

    `write_replay=False` writes only the sidecars, for a replay converted before they existed:
    the replay is left as it is, and its `source.json` only gains `extras`. It must be the
    replay of this log (same duration), or this raises."""
    log_path, out_dir = Path(log_path), Path(out_dir)
    stem = stem or log_stem(log_path.name)
    scan = _Scan(log_path, raw_vision=raw_vision)
    if not scan.referee:
        raise ValueError(f"{log_path}: no referee messages")

    # from the first message of a match stage to the first after it that ends play
    first = next((i for i, (_, msg) in enumerate(scan.referee) if msg.stage not in _END_STAGES), None)
    if first is None:
        raise ValueError(f"{log_path}: no referee message before the end of play")
    start_ns = scan.referee[first][0]
    end_ns = next((t for t, msg in scan.referee[first:] if msg.stage in _END_STAGES), scan.last_ns)
    referee = [(t, msg) for t, msg in scan.referee[first:] if t < end_ns]
    ts = np.arange(0.0, (end_ns - start_ns) / 1e9, TICK_S)
    tick_ns = start_ns + np.round(ts * 1e9).astype(np.int64)

    ref_index = np.searchsorted(np.array([t for t, _ in referee], dtype=np.int64), tick_ns, side="right") - 1
    swapped_msg = _colours_swapped(referee)
    swapped = swapped_msg[ref_index]

    tracker_frames = {name: len(frames) for name, frames in scan.trackers.items()}
    if scan.trackers:
        source = max(scan.trackers, key=lambda name: (len(scan.trackers[name]), name))
        world, dropped_ids = _tracker_world(scan.trackers[source], tick_ns)
    else:
        source = "vision"
        vision = scan.vision[VISION_2014] or scan.vision[VISION_2010]
        world, dropped_ids = _vision_world(vision, tick_ns)
    _to_kick_off_colours(world, swapped)
    field = _geometry(scan.geometry)
    walls = (
        (field["field_length"] / 2 + field["boundary_width"], field["field_width"] / 2 + field["boundary_width"])
        if field
        else _DIV_B_WALLS
    )
    world.ball_seen &= (np.abs(world.ball_p[:, 0]) <= walls[0]) & (np.abs(world.ball_p[:, 1]) <= walls[1])
    world.finish()

    stage = np.array([referee[i][1].stage for i in ref_index])
    keepers = np.array([(msg.blue if s else msg.yellow).goalkeeper for (_, msg), s in zip(referee, swapped_msg)])
    keepers = keepers[ref_index]
    blue_right = [
        msg.blue_team_on_positive_half if msg.HasField("blue_team_on_positive_half") else None for _, msg in referee
    ]
    side_is_right = np.array([blue_right[i] is bool(swapped_msg[i]) for i in ref_index])
    unknown = np.array([blue_right[i] is None for i in ref_index])
    if unknown.any():
        side_is_right[unknown] = _sides_from_goalkeepers(world, stage, keepers)[unknown]

    command = np.array([referee[i][1].command for i in ref_index])
    counter = np.array([referee[i][1].command_counter for i in ref_index])
    taken = free_kick_taken(command, counter, world.ball_p[:, :2])
    live = taken | np.isin(command, (RefereeCommand.NORMAL_START.value, RefereeCommand.FORCE_START.value))
    inverted = _contradicted_sides(world, side_is_right, live, keepers)
    side_is_right ^= inverted

    out_dir.mkdir(parents=True, exist_ok=True)
    duration_s = round(float(ts[-1]) if len(ts) else 0.0, 3)
    source_path = out_dir / f"{stem}.source.json"
    if not write_replay:
        summary = json.loads(source_path.read_text())
        if summary["duration_s"] != duration_s:
            raise ValueError(f"{source_path}: {summary['duration_s']} s, but {log_path.name} gives {duration_s} s")
        summary["extras"] = _write_extras(out_dir, stem, world, source, scan, start_ns, end_ns, tick_ns, swapped)
        source_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    seen = ~np.isnan(world.robot_p[..., 0])
    writer = ColumnarReplayWriter(
        ColumnarReplayWriterConfig(replay_name=stem, overwrite_existing=True, checkpoint_every_s=0),
        my_team_is_yellow=True,
        exp_friendly=int(seen[:, _YELLOW].sum(axis=1).max(initial=0)),
        exp_enemy=int(seen[:, _BLUE].sum(axis=1).max(initial=0)),
        path=out_dir / f"{stem}.npz",
    )
    # each tick's referee is the last kept message with its clocks run on from when it arrived
    base_index, base, base_ts = -1, None, 0.0
    for k, i in enumerate(ref_index):
        if i != base_index:
            base_index, base_ts = i, (referee[i][0] - start_ns) / 1e9
            base = referee_data(referee[i][1], referee[i][0] / 1e9, swapped=bool(swapped_msg[i]))
            if inverted[k] and base.blue_team_on_positive_half is not None:  # see `_contradicted_sides`
                base = dataclasses.replace(base, blue_team_on_positive_half=not base.blue_team_on_positive_half)
        if taken[k] and base.referee_command != RefereeCommand.NORMAL_START:  # see `free_kick_taken`
            base = dataclasses.replace(
                advance_clocks(base, float(ts[k] - base_ts)),
                referee_command=RefereeCommand.NORMAL_START,
                referee_command_timestamp=tick_ns[k] / 1e9,
            )
            base_ts = ts[k]
        ball = Ball(p=Vector3D(*world.ball_p[k]), v=Vector3D(*world.ball_v[k]), a=Vector3D(*world.ball_a[k]))
        writer.write_frame(
            GameFrame(
                ts=float(ts[k]),
                my_team_is_yellow=True,
                my_team_is_right=bool(side_is_right[k]),
                friendly_robots=_robots(world, k, _YELLOW, True),
                enemy_robots=_robots(world, k, _BLUE, False),
                ball=ball,
                referee=advance_clocks(base, float(ts[k] - base_ts)),
            )
        )
    writer.close()

    last = referee_data(referee[-1][1], referee[-1][0] / 1e9, swapped=bool(swapped_msg[-1]))
    summary = {
        "log": log_path.name,
        "yellow": last.yellow_team.name,
        "blue": last.blue_team.name,
        "score": {"yellow": last.yellow_team.score, "blue": last.blue_team.score},
        "colours_swapped": bool(swapped_msg.any()),
        "start_unix_s": start_ns / 1e9,
        "duration_s": duration_s,
        "world_source": source,
        "tracker_frames": tracker_frames,
        "ball_unseen_share": round(float(1 - world.ball_seen.mean()) if len(ts) else 1.0, 4),
        "robot_ids_dropped": dropped_ids,
        "sides_from_goalkeepers": bool(unknown.any()),
        "sides_inverted_share": round(float(inverted.mean()) if len(ts) else 0.0, 4),
        "field": field,
        "extras": _write_extras(out_dir, stem, world, source, scan, start_ns, end_ns, tick_ns, swapped),
    }
    source_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _to_kick_off_colours(world: _World, swapped: np.ndarray) -> None:
    for robots in (
        world.robot_p,
        world.robot_v,
        world.robot_orientation,
        world.robot_visibility,
        world.robot_vel_angular,
    ):
        robots[swapped] = robots[swapped][:, ::-1]


def _extras_source(frames: Counter, source: str) -> Optional[str]:
    """The tracker source to take kicks or visibility from: the replay's own if it sends them,
    else the one that sends them in the most frames; None if none does."""
    if frames[source] > 0:
        return source
    best = max(frames, key=lambda name: (frames[name], name), default=None)
    return best if best is not None and frames[best] > 0 else None


def _write_extras(
    out_dir: Path, stem: str, world: _World, source: str, scan: _Scan, start_ns, end_ns, tick_ns, swapped
) -> dict:
    """Write the sidecars (module docstring) this log has data for; returns `source.json`'s
    `extras`, which names the tracker source the kicks and the visibility came from. The
    replay's tracker (`world`, from `source`) may not send them; another autoref's may."""
    extras = {"kicks": None, "kicks_source": None, "visibility_source": None, "tracker": False, "raw_vision": False}
    if scan.trackers:
        worlds = {source: world}

        def world_of(name: Optional[str]) -> _World:
            if name is None:
                return world  # NaN visibility and no kicks, as sent
            if name not in worlds:
                worlds[name] = _tracker_world(scan.trackers[name], tick_ns)[0]
                _to_kick_off_colours(worlds[name], swapped)
            return worlds[name]

        kicks_source = _extras_source(scan.kick_frames, source)
        visibility_source = _extras_source(scan.visibility_frames, source)
        kicks = _kick_rows(world_of(kicks_source).kicks, start_ns, end_ns, tick_ns, swapped)
        (out_dir / f"{stem}.kicks.json").write_text(json.dumps(kicks, indent=1) + "\n")
        shown = world_of(visibility_source)
        _save_arrays(
            out_dir / f"{stem}.tracker.zip",
            ts=(tick_ns - start_ns) / 1e9,
            ball_visibility=shown.ball_visibility.astype(np.float32),
            robot_visibility=shown.robot_visibility.astype(np.float32),
            robot_vel_angular=world.robot_vel_angular.astype(np.float32),
        )
        extras.update(kicks=len(kicks), kicks_source=kicks_source, visibility_source=visibility_source, tracker=True)
    if scan.raw_vision is not None:
        _save_arrays(out_dir / f"{stem}.vision.zip", **scan.raw_vision.arrays(start_ns, end_ns, tick_ns, swapped))
        extras["raw_vision"] = True
    return extras


def _save_arrays(path: Path, **arrays: np.ndarray) -> None:
    """`np.savez_compressed` under `path` exactly (it appends `.npz` to a name that lacks it)."""
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def _kick_rows(kicks: list[dict], start_ns: int, end_ns: int, tick_ns: np.ndarray, swapped: np.ndarray) -> list[dict]:
    """The tracker's kicks (`_add_kick`) that started inside the replay window, as `.kicks.json` rows."""
    rows = []
    for kick in kicks:
        if not start_ns <= kick["start_ns"] < end_ns:
            continue
        team = rid = None
        if kick["kicker"] is not None and kick["kicker"][0] in (_YELLOW, _BLUE):
            tick = max(int(np.searchsorted(tick_ns, kick["start_ns"], side="right")) - 1, 0)
            team = ("yellow", "blue")[kick["kicker"][0] ^ int(swapped[tick])]
            rid = kick["kicker"][1]
        row = {
            "ts": round((kick["start_ns"] - start_ns) / 1e9, 4),
            "team": team,
            "robot_id": rid,
            "pos": [round(v, 4) for v in kick["pos"]],
            "vel_first": [round(v, 4) for v in kick["vel_first"]],
            "vel": [round(v, 4) for v in kick["vel"]],
            "speed": round(math.hypot(*kick["vel"]), 4),
            "reports": kick["reports"],
            "kickers": len(kick["kickers"]),
        }
        if "stop_ns" in kick:
            row["stop_ts"] = round((kick["stop_ns"] - start_ns) / 1e9, 4)
        if "stop_pos" in kick:
            row["stop_pos"] = [round(v, 4) for v in kick["stop_pos"]]
        rows.append(row)
    return rows
