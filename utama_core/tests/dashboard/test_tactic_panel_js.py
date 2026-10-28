"""`status_panel.js`'s Tactics panel run in node: each robot's tactic, or the referee's job for it,
with the steps its tactic traced on a line underneath. Skipped where node is missing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).parents[2] / "dashboard" / "static" / "status_panel.js").read_text()
# `TACTIC_TAG_COLORS` comes from field_canvas.js on the page; a stand-in is enough here.
PRELUDE = """
const TACTIC_TAG_COLORS = { attack: "#d9694f", referee: "#c8c8c8" };
const panel = { innerHTML: "" };
const document = { getElementById: () => panel };
"""

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def _panel(status) -> str:
    js = (
        PRELUDE + SCRIPT + f"\nrenderTacticStatusInto('x', {json.dumps(status)});process.stdout.write(panel.innerHTML);"
    )
    return subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout


def test_a_robots_steps_show_under_its_tactic():
    html = _panel({"3": ["give_and_go", "attack", "shoot: turning to shoot"]})
    assert "give_and_go" in html
    assert 'tactic-steps">shoot: turning to shoot</div>' in html


def test_a_robot_the_referee_drives_shows_its_job():
    html = _panel({"2": ["referee · ball placement · placer", "referee", ""]})
    assert "referee · ball placement · placer" in html
    assert "tactic-steps" not in html  # no steps, no empty line


def test_steps_are_escaped():
    html = _panel({"1": ["t", "attack", "a<b>"]})
    assert "a&lt;b&gt;" in html
