"""Driving a robot from the Live tab's Teleop panel (`tactics/teleop.py`): the robot plays the
page's latest command, stops once the page goes quiet for `DEADMAN_S`, kicks once per press,
and never goes faster than the caps; a robot the run does not have is refused."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from utama_core.dashboard.views import teleop as teleop_view
from utama_core.tactics.teleop import (
    DEADMAN_S,
    MAX_SPEED,
    MAX_TURN,
    TeleopDrive,
    TeleopTactic,
)


def _drive():
    now = [0.0]
    return TeleopDrive(["yellow 0", "blue 0"], clock=lambda: now[0]), now


def test_the_robot_plays_the_latest_command_until_the_page_goes_quiet():
    drive, now = _drive()
    drive.set("yellow 0", forward=0.3, turn=1.5, dribble=True)
    now[0] = DEADMAN_S - 0.01
    assert drive.command("yellow 0")[:3] == (0.3, 0.0, 1.5) and drive.command("yellow 0").dribble
    assert drive.command("blue 0")[:3] == (0.0, 0.0, 0.0)  # never sent anything
    now[0] = DEADMAN_S + 0.01
    assert drive.command("yellow 0")[:3] == (0.0, 0.0, 0.0) and not drive.command("yellow 0").dribble


def test_a_kick_is_played_once_even_when_the_next_message_comes_first():
    drive, _ = _drive()
    drive.set("yellow 0", kick=True)
    drive.set("yellow 0", forward=0.3)  # the next send, before a tick played the kick
    assert drive.command("yellow 0").kick
    assert not drive.command("yellow 0").kick


def test_commands_are_capped_and_an_unknown_robot_refused():
    drive, _ = _drive()
    drive.set("blue 0", forward=5.0, left=-5.0, turn=-9.0)
    assert drive.command("blue 0")[:3] == (MAX_SPEED, -MAX_SPEED, -MAX_TURN)
    with pytest.raises(ValueError, match="no robot"):
        drive.set("yellow 3", forward=0.1)


def test_the_tactic_drives_its_robot_and_the_view_reaches_the_drive():
    drive, _ = _drive()
    routes, handlers = {}, {}
    teleop_view.attach(SimpleNamespace(add_route=routes.__setitem__, add_command_handler=handlers.__setitem__), drive)
    assert json.loads(routes["/teleop/config"]()) == {"robots": ["yellow 0", "blue 0"]}
    handlers["/teleop/drive"]({"robot": "yellow 0", "left": 0.3})
    commands, _ = TeleopTactic(drive, "yellow 0").tick(None, None, (4, 5), None)
    assert commands[4][:3] == (0.0, 0.3, 0.0) and commands[5][:3] == (0.0, 0.0, 0.0)
