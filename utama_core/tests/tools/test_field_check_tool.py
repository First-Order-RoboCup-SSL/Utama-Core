"""`tools/field_check.py` flags a slow camera, a missing ball or lids, and no vision at all; the
robots part flags a robot answering few of its commands."""

import collections
import importlib.util
from pathlib import Path

from utama_core.team_controller.src.controllers.real import find_robots as fr

spec = importlib.util.spec_from_file_location(
    "field_check", Path(__file__).resolve().parents[3] / "tools" / "field_check.py"
)
field_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(field_check)


def heard(per_camera, things=("ball", "yellow 0"), field=(9000, 6000)):
    return {
        "senders": {"10.0.0.2"} if per_camera else set(),
        "per_camera": collections.Counter(per_camera),
        "field": field,
        "positions": {(t, 0): [] for t in things},
    }


def test_a_camera_below_min_hz_and_a_missing_ball_are_flagged():
    seconds = 2.0
    lines = field_check.vision_lines(
        heard({0: 60 * seconds, 1: (field_check.MIN_CAMERA_HZ - 1) * seconds}, ("yellow 0",)), seconds
    )
    assert lines[0].startswith("ok camera 0") and lines[1].startswith("!! camera 1")
    assert "!! no ball in view" in lines and "ok lids in view: yellow 0" in lines


def test_no_vision_is_one_line_saying_so():
    (line,) = field_check.vision_lines(heard({}), 2.0)
    assert line.startswith("!! vision: nothing")


def test_a_robot_answering_few_commands_is_reported():
    row = {"radio": 0, "answered": True, "lid": ["yellow", 0], "turned": ["yellow 0"]}
    found = {"time": "t", "lids_seen": ["yellow 0"], "robots": [{**row, "reply_rate": fr.LOW_REPLY_RATE - 0.01}]}
    assert "Few replies (radio 0" in fr.report(found)
    found["robots"][0]["reply_rate"] = fr.LOW_REPLY_RATE
    assert "Few replies" not in fr.report(found)
