"""What crosses the process boundary of a `.strategy` file: length-prefixed JSON.

Standard library only, and it reads frames by attribute: this file ships inside every bundle and
must work with the entity classes of whichever commit the bundle holds. JSON and not pickle because
the runner must be able to read what a bundle sends without running anything it says. A Python float
survives JSON exactly, so a bundled match can equal the same match run in one process.
"""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO, Optional

_HEADER = struct.Struct("<I")


def send(stream: BinaryIO, message: Any) -> None:
    data = json.dumps(message, separators=(",", ":")).encode()
    stream.write(_HEADER.pack(len(data)) + data)
    stream.flush()


def receive(stream: BinaryIO) -> Optional[Any]:
    """The next message, or None when the other side has gone."""
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        return None
    (length,) = _HEADER.unpack(header)
    data = stream.read(length)
    if len(data) < length:
        return None
    return json.loads(data)


def _vector(vector) -> list[float]:
    return [float(vector.x), float(vector.y)] + ([float(vector.z)] if hasattr(vector, "z") else [])


def _robot(robot) -> list:
    return [
        int(robot.id),
        bool(robot.is_friendly),
        bool(robot.has_ball),
        _vector(robot.p),
        _vector(robot.v),
        _vector(robot.a),
        float(robot.orientation),
    ]


def _team(team) -> dict:
    return dict(vars(team))


def _referee(referee) -> Optional[dict]:
    if referee is None:
        return None
    return {
        "source_identifier": referee.source_identifier,
        "time_sent": referee.time_sent,
        "time_received": referee.time_received,
        "referee_command": referee.referee_command.name,
        "referee_command_timestamp": referee.referee_command_timestamp,
        "stage": referee.stage.name,
        "stage_time_left": referee.stage_time_left,
        "blue_team": _team(referee.blue_team),
        "yellow_team": _team(referee.yellow_team),
        "designated_position": referee.designated_position,
        "blue_team_on_positive_half": referee.blue_team_on_positive_half,
        "next_command": None if referee.next_command is None else referee.next_command.name,
        "current_action_time_remaining": referee.current_action_time_remaining,
        "match_type": referee.match_type,
        "status_message": referee.status_message,
    }


def encode_frame(frame) -> dict:
    """A `GameFrame`, without the referee's raw `game_events`: no strategy reads them."""
    ball = frame.ball
    return {
        "ts": float(frame.ts),
        "my_team_is_yellow": bool(frame.my_team_is_yellow),
        "my_team_is_right": bool(frame.my_team_is_right),
        "friendly_robots": [_robot(robot) for robot in frame.friendly_robots.values()],
        "enemy_robots": [_robot(robot) for robot in frame.enemy_robots.values()],
        "ball": None if ball is None else [_vector(ball.p), _vector(ball.v), _vector(ball.a)],
        "referee": _referee(frame.referee),
    }


def encode_field(field) -> dict:
    dims = field._field_dims
    bounds = field._field_bounds
    return {
        "my_team_is_right": field.my_team_is_right,
        "dims": {name: getattr(dims, name) for name in dims.__dataclass_fields__},
        "top_left": list(bounds.top_left),
        "bottom_right": list(bounds.bottom_right),
        "defense_area_rules": field._defense_area_rules,
    }
