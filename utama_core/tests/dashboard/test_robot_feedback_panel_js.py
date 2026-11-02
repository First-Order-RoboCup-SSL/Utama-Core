"""`status_panel.js`'s robot panel run in node: each radio robot's ball sensor, radio age and
position, as the Live tab shows them on real robots. Skipped where node is missing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).parents[2] / "dashboard" / "static" / "status_panel.js").read_text()

ROBOTS = {
    "friendly": [{"id": 0, "x": 0.5, "y": -0.25, "vx": 0.3, "vy": 0.4}],
    "enemy": [{"id": 1, "x": -1.0, "y": 0.0, "vx": 0.0, "vy": 0.0}],
}


def _html(feedback, ball_last_on=None, robots=ROBOTS) -> str:
    d = {"robot_feedback": feedback, "robots": robots}
    js = SCRIPT + f"\nprocess.stdout.write(robotFeedbackHtml({json.dumps(d)}, {json.dumps(ball_last_on or {})}));"
    return subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout


def _row(port_id, vision_id, team, colour, *, has_ball=False, connected=True, age=0.02):
    return {
        "port_id": port_id,
        "vision_id": vision_id,
        "team": team,
        "team_color": colour,
        "has_ball": has_ball,
        "connected": connected,
        "age_seconds": age,
    }


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def test_sensor_on_shows_ball_with_radio_age_and_position():
    html = _html([_row(1, 0, "friendly", "blue", has_ball=True)])
    assert "blue 0" in html and "BALL" in html
    assert "fw 1 · 20 ms" in html
    assert "(0.50, -0.25) · 0.50 m/s" in html


def test_sensor_off_shows_when_it_was_last_on():
    html = _html([_row(1, 0, "friendly", "blue")], ball_last_on={"1": 2.34})
    assert "BALL" not in html and "last 2.3 s ago" in html


def test_a_silent_radio_does_not_report_the_sensor():
    # A stale has_ball must not read as "the robot has the ball".
    html = _html([_row(0, 1, "enemy", "yellow", has_ball=True, connected=False, age=3.2)])
    assert "BALL" not in html and "ball sensor ?" in html
    assert "fw 0 silent 3.2 s" in html


def test_never_answered_and_not_seen():
    html = _html([_row(0, 1, "enemy", "yellow", connected=False, age=None)], robots={"friendly": [], "enemy": []})
    assert "fw 0 no feedback yet" in html and "not seen" in html


def test_no_feedback_rows_draws_nothing():
    assert _html([]) == ""


def test_the_share_of_commands_answered_shows_and_turns_red_below_the_low_rate():
    good = {**_row(1, 0, "friendly", "blue"), "reply_rate": 0.97}
    poor = {**_row(2, 0, "enemy", "yellow"), "reply_rate": 0.79}
    html = _html([good, poor])
    assert "fw 1 · 20 ms · 97% answered" in html
    assert '<span class="check-fail">79% answered</span>' in html
    assert "answered" not in _html([_row(1, 0, "friendly", "blue")])  # a run that counts none
