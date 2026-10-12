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
- Perspective: yellow is the friendly team (`my_team_is_yellow`), blue the enemy.
  `side_is_right[t]` (yellow defends the +x goal) is the referee's `blue_team_on_positive_half`
  negated; for a message without that field (logs before 2019) it is where yellow's goalkeeper
  stood during that stage.
- Time: 60 Hz ticks on the log's receive clock, `ts` 0 at the first referee message. The
  window ends where play ends: a penalty shoot-out and the post-game stage are dropped. Breaks,
  timeouts and halts stay in, so `ts` has no gaps; skip them by `stage`/`referee_command`.
- Robots and ball: from the tracker, an autoref's filtered world in metres with velocities, the
  source with the most frames when several autorefs ran. A tick takes the newest tracked frame
  at or before it; after `_STALE_S` without one, robots are absent. Logs without a tracker
  (before 2020) merge the cameras' raw detections instead and difference positions over
  `_DIFF_TICKS` ticks for velocities. Accelerations are differenced velocities in both cases.
  The ball holds its last position (at rest) while no ball is seen.
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
- `<stem>.source.json`: the log, teams, final score, tracker source and field geometry.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
import struct
import urllib.parse
from collections import defaultdict
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


def referee_data(msg: Referee, received_s: float) -> RefereeData:
    """`msg` as `RefereeData`, field for field as `RefereeMessageReceiver._update_data`, but with
    the protocol's units converted: `stage_time_left` us -> s, `designated_position` mm -> m."""
    return RefereeData(
        source_identifier=msg.source_identifier,
        time_sent=msg.packet_timestamp / 1e6,
        time_received=received_s,
        referee_command=RefereeCommand.from_id(msg.command),
        referee_command_timestamp=msg.command_timestamp / 1e6,
        stage=Stage.from_id(msg.stage),
        stage_time_left=msg.stage_time_left / 1e6,
        blue_team=_team_info("blue", msg.blue),
        yellow_team=_team_info("yellow", msg.yellow),
        designated_position=(
            (msg.designated_position.x / 1000, msg.designated_position.y / 1000)
            if msg.HasField("designated_position")
            else None
        ),
        blue_team_on_positive_half=(
            msg.blue_team_on_positive_half if msg.HasField("blue_team_on_positive_half") else None
        ),
        next_command=RefereeCommand.from_id(msg.next_command) if msg.HasField("next_command") else None,
        current_action_time_remaining=(
            msg.current_action_time_remaining if msg.HasField("current_action_time_remaining") else None
        ),
        game_events=[MessageToDict(e, preserving_proto_field_name=True) for e in msg.game_events],
        match_type=msg.match_type,
        status_message=msg.status_message or None,
    )


class _Scan:
    """One pass over a log: the referee messages that change something, every tracker source's
    frames, the camera detections and the field geometry."""

    def __init__(self, path: Path):
        self.referee: list[tuple[int, Referee]] = []
        self.trackers: dict[str, list[tuple[int, bytes]]] = defaultdict(list)
        self.vision: dict[int, list[tuple[int, bytes]]] = defaultdict(list)  # by record type
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
                    self.trackers[packet.source_name or packet.uuid].append((ts_ns, payload))
                    self.vision.clear()
            elif msg_type in (VISION_2010, VISION_2014):
                if not self.trackers:  # camera detections are only read when there is no tracker
                    self.vision[msg_type].append((ts_ns, payload))
                if msg_type == VISION_2014 and self.geometry is None:
                    packet = SSL_WrapperPacket()
                    packet.ParseFromString(payload)
                    if packet.HasField("geometry"):
                        self.geometry = packet.geometry.field
            else:
                continue  # the index record (type 6) has no meaningful time
            self.last_ns = max(self.last_ns, ts_ns)


class _World:
    """Robots and ball on the tick grid. Robot arrays are [tick, team, id, ...], NaN where absent."""

    def __init__(self, n_ticks: int):
        self.robot_p = np.full((n_ticks, 2, _MAX_ID, 2), np.nan)
        self.robot_v = np.full((n_ticks, 2, _MAX_ID, 2), np.nan)
        self.robot_orientation = np.full((n_ticks, 2, _MAX_ID), np.nan)
        self.ball_p = np.full((n_ticks, 3), np.nan)
        self.ball_v = np.full((n_ticks, 3), np.nan)
        self.ball_seen = np.zeros(n_ticks, dtype=bool)

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
        if i != decoded:
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
        if frame.balls:
            ball = frame.balls[0]  # the primary ball
            world.ball_p[k] = (ball.pos.x, ball.pos.y, ball.pos.z)
            world.ball_seen[k] = True
            if ball.HasField("vel"):
                world.ball_v[k] = (ball.vel.x, ball.vel.y, ball.vel.z)
    return world, dropped


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


def _sides_from_goalkeepers(world: _World, stage: np.ndarray, keepers: np.ndarray) -> np.ndarray:
    """Per tick whether yellow defends +x, for logs whose referee does not say: within each
    stage, the side of the median x of yellow's goalkeeper (else of yellow's robots relative to
    blue's). A stage with no robots seen keeps the previous stage's answer."""
    sides = np.ones(len(stage), dtype=bool)
    previous = True
    for value in np.unique(stage):
        ticks = np.flatnonzero(stage == value)
        keeper_x = world.robot_p[ticks, _YELLOW, np.minimum(keepers[ticks], _MAX_ID - 1), 0]
        keeper_x = keeper_x[~np.isnan(keeper_x)]
        if len(keeper_x):
            previous = bool(np.median(keeper_x) > 0)
        else:
            yellow = world.robot_p[ticks, _YELLOW, :, 0]
            blue = world.robot_p[ticks, _BLUE, :, 0]
            if np.isfinite(yellow).any() and np.isfinite(blue).any():
                previous = bool(np.nanmean(yellow) > np.nanmean(blue))
        sides[ticks] = previous
    return sides


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


def convert_log(log_path, out_dir, stem: Optional[str] = None) -> dict:
    """Convert one game log to `<out_dir>/<stem>.npz` (+ `.sparse_referee.pkl`, `.source.json`).
    Returns what `source.json` holds. See the module docstring for what the replay contains."""
    log_path, out_dir = Path(log_path), Path(out_dir)
    stem = stem or log_stem(log_path.name)
    scan = _Scan(log_path)
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

    tracker_frames = {name: len(frames) for name, frames in scan.trackers.items()}
    if scan.trackers:
        source = max(scan.trackers, key=lambda name: (len(scan.trackers[name]), name))
        world, dropped_ids = _tracker_world(scan.trackers[source], tick_ns)
    else:
        source = "vision"
        vision = scan.vision[VISION_2014] or scan.vision[VISION_2010]
        world, dropped_ids = _vision_world(vision, tick_ns)
    world.finish()

    ref_index = np.searchsorted(np.array([t for t, _ in referee], dtype=np.int64), tick_ns, side="right") - 1
    stage = np.array([referee[i][1].stage for i in ref_index])
    blue_right = [
        msg.blue_team_on_positive_half if msg.HasField("blue_team_on_positive_half") else None for _, msg in referee
    ]
    side_is_right = np.array([blue_right[i] is False for i in ref_index])
    unknown = np.array([blue_right[i] is None for i in ref_index])
    if unknown.any():
        keepers = np.array([referee[i][1].yellow.goalkeeper for i in ref_index])
        side_is_right[unknown] = _sides_from_goalkeepers(world, stage, keepers)[unknown]

    command = np.array([referee[i][1].command for i in ref_index])
    counter = np.array([referee[i][1].command_counter for i in ref_index])
    taken = free_kick_taken(command, counter, world.ball_p[:, :2])

    out_dir.mkdir(parents=True, exist_ok=True)
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
            base = referee_data(referee[i][1], referee[i][0] / 1e9)
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

    last = referee[-1][1]
    summary = {
        "log": log_path.name,
        "yellow": last.yellow.name,
        "blue": last.blue.name,
        "score": {"yellow": last.yellow.score, "blue": last.blue.score},
        "start_unix_s": start_ns / 1e9,
        "duration_s": round(float(ts[-1]) if len(ts) else 0.0, 3),
        "world_source": source,
        "tracker_frames": tracker_frames,
        "ball_unseen_share": round(float(1 - world.ball_seen.mean()) if len(ts) else 1.0, 4),
        "robot_ids_dropped": dropped_ids,
        "sides_from_goalkeepers": bool(unknown.any()),
        "field": _geometry(scan.geometry),
    }
    (out_dir / f"{stem}.source.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
