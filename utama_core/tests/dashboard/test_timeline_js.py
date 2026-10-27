"""`timeline.js`'s match clock run in node: the replay showed only its own time, which keeps
running through stoppages, so a 10 min match read 12 min with nothing to match the referee's
clock against. Skipped where node is missing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).parents[2] / "dashboard" / "static" / "timeline.js").read_text()

REFEREE = [
    {"sim_time": 0, "command": "PREPARE_KICKOFF_YELLOW", "stage": "NORMAL_FIRST_HALF_PRE"},
    {"sim_time": 5, "command": "NORMAL_START", "stage": "NORMAL_FIRST_HALF"},
    {"sim_time": 8, "command": "STOP", "stage": "NORMAL_FIRST_HALF"},
    {"sim_time": 9, "command": "BALL_PLACEMENT_BLUE", "stage": "NORMAL_FIRST_HALF"},
    {"sim_time": 10, "command": "DIRECT_FREE_BLUE", "stage": "NORMAL_FIRST_HALF"},
    {"sim_time": 12, "command": "NORMAL_START", "stage": "NORMAL_FIRST_HALF"},
    {"sim_time": 20, "command": "STOP", "stage": "NORMAL_HALF_TIME"},
]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_match_clock_runs_only_in_play_and_restarts_each_stage():
    js = (
        "const module = { exports: {} };\n"
        + SCRIPT
        + f"\nconst clock = module.exports.matchClock({json.dumps(REFEREE)});"
        + "\nconsole.log(JSON.stringify([-1, 6, 9.5, 11, 15, 21].map(clock)));"
    )
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out[0] is None
    # 5-8 in play, 8-10 stopped (STOP, placement), 10-20 in play (the free kick counts)
    assert [c["played"] for c in out[1:]] == pytest.approx([1, 3, 4, 8, 0])
    assert out[-1]["stage"] == "NORMAL_HALF_TIME"
