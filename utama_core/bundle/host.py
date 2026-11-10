"""The process a `.strategy` file runs in: game frames in on stdin, robot commands out on stdout.

Run as `python -m utama_core.bundle.host` from an unpacked bundle, so `utama_core` is the bundle's own
copy. This file and `wire.py` are the current ones even in a bundle of an older commit; the rest is
that commit's code.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

from utama_core.bundle import wire
from utama_core.config.enums import Mode
from utama_core.config.field_params import FieldBounds, FieldDimensions
from utama_core.config.settings import MAX_GAME_HISTORY
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_log import MatchLog
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.motion_planning.src.common.control_schemes import CONTROL_SCHEME_MAP


class _CommandRecorder:
    """Stands in for the robot controller: `AbstractStrategy.step` fills it, the host sends it."""

    keeps_kickers_charged = False

    def __init__(self) -> None:
        self.commands: dict[str, list] = {}

    def add_robot_commands(self, command, robot_id: int) -> None:
        self.commands[str(robot_id)] = [
            float(command.local_forward_vel),
            float(command.local_left_vel),
            float(command.angular_vel),
            bool(command.kick),
            bool(command.chip),
            bool(command.dribble),
        ]

    def send_robot_commands(self) -> None:
        pass


def _robot(values: list) -> Robot:
    rid, is_friendly, has_ball, p, v, a, orientation = values
    return Robot(rid, is_friendly, has_ball, Vector2D(*p), Vector2D(*v), Vector2D(*a), orientation)


def _referee(data: dict | None) -> RefereeData | None:
    if data is None:
        return None
    next_command = data.pop("next_command")
    return RefereeData(
        **{
            **data,
            "referee_command": RefereeCommand[data["referee_command"]],
            "stage": Stage[data["stage"]],
            "blue_team": TeamInfo(**data["blue_team"]),
            "yellow_team": TeamInfo(**data["yellow_team"]),
            "designated_position": None if data["designated_position"] is None else tuple(data["designated_position"]),
            "next_command": None if next_command is None else RefereeCommand[next_command],
        }
    )


def decode_frame(data: dict) -> GameFrame:
    ball = data["ball"]
    return GameFrame(
        ts=data["ts"],
        my_team_is_yellow=data["my_team_is_yellow"],
        my_team_is_right=data["my_team_is_right"],
        friendly_robots={values[0]: _robot(values) for values in data["friendly_robots"]},
        enemy_robots={values[0]: _robot(values) for values in data["enemy_robots"]},
        ball=None if ball is None else Ball(Vector3D(*ball[0]), Vector3D(*ball[1]), Vector3D(*ball[2])),
        referee=_referee(data["referee"]),
    )


def decode_field(data: dict) -> Field:
    return Field(
        data["my_team_is_right"],
        FieldDimensions(**data["dims"]),
        FieldBounds(tuple(data["top_left"]), tuple(data["bottom_right"])),
        data["defense_area_rules"],
    )


def _drain(log) -> list:
    """What the strategy logged since the last call, as `[event class name, fields]`."""
    if log is None:
        return []
    out = []
    for event in log._events:
        fields = dataclasses.asdict(event)
        if "tag" in fields:
            fields["tag"] = fields["tag"].value
        out.append([type(event).__name__, fields])
    log._events.clear()
    return out


def _slot_status(strategy) -> dict:
    return {
        tactic_id: {"robots": list(info["robots"]), "committed": bool(info["committed"]), "tag": info["tag"].value}
        for tactic_id, info in strategy._kernel_strategy.slot_status(strategy.game).items()
    }


def _partition(strategy) -> dict:
    kernel = strategy._kernel_strategy
    return {
        "partition": {tactic_id: list(robots) for tactic_id, robots in kernel.active_partition.items()},
        "names": {tactic_id: type(kernel._tactics[tactic_id]).__name__ for tactic_id in kernel.active_partition},
    }


_QUERIES = {
    "slot_status": _slot_status,
    "partition": _partition,
    "debug_status": lambda strategy: {str(rid): row for rid, row in strategy.debug_status().items()},
}


def _controller(setup: dict):
    """The motion controller as the runner configured its own."""
    scheme = next(c for c in CONTROL_SCHEME_MAP.values() if c.__name__ == setup["controller"])
    controller = scheme(Mode(setup["mode"]), None)
    controller.pose_ahead.latency_s = setup["latency_s"]
    controller.robot_carry = {int(rid): tuple(carry) for rid, carry in setup["robot_carry"].items()}
    return controller


def _builder(manifest: dict):
    """`AbstractStrategy`'s first argument. The file's own code decides how: files of commits from
    before `strategy/catalog.py` have `kernel_strategy.build_<name>_kernel_strategy` instead."""
    ids = tuple(manifest["outfield_ids"])
    name = manifest["strategy"]
    try:
        from utama_core.strategy.catalog import builder
    except ModuleNotFoundError:
        from utama_core.strategy import kernel_strategy

        return getattr(kernel_strategy, f"build_{name}_kernel_strategy")(ids)
    return builder(name, ids)


def main() -> None:
    # Strategy code that prints must not corrupt the channel.
    replies = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    requests = sys.stdin.buffer

    with open("manifest.json") as f:
        manifest = json.load(f)
    strategy = AbstractStrategy(_builder(manifest))
    recorder = _CommandRecorder()
    strategy.load_robot_controller(recorder)
    wire.send(replies, {"control_scheme": strategy.control_scheme})

    log = None
    while (message := wire.receive(requests)) is not None:
        kind = message["type"]
        if kind == "setup":
            strategy.load_motion_controller(_controller(message))
        elif kind == "game":
            history = GameHistory(MAX_GAME_HISTORY)
            strategy.load_game(Game(history, decode_frame(message["frame"]), field=decode_field(message["field"])))
        elif kind == "match_log":
            log = MatchLog()
            strategy._kernel_strategy.match_log = log
        elif kind == "frame":
            strategy.game.add_game_frame(decode_frame(message["frame"]))
        elif kind == "tick":
            recorder.commands = {}
            strategy.step()
            wire.send(replies, {"commands": recorder.commands, "events": _drain(log)})
        elif kind == "query":
            wire.send(replies, _QUERIES[message["name"]](strategy))


if __name__ == "__main__":
    main()
