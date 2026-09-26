"""Playback regression for the OpenJev HTML replay page's JS playhead."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from tools.jev_replay_html import PAGE


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page JS")
@pytest.mark.parametrize("refresh_hz", [60, 120])
def test_playhead_advances_when_display_refresh_is_faster_than_frames(refresh_hz):
    # stride 2 -> 30 fps frames; one second of display refreshes must play ~30 frames,
    # not stall on frame 0 because each refresh is shorter than one frame.
    m = re.search(r"^function advance\(.*$", PAGE, re.M)
    assert m, "replay page lost its advance() playhead function"
    advance = m.group(0)
    ts = [k / 30 for k in range(90)]
    js = f"""{advance}
let i = 0, pt = 0; const ts = {json.dumps(ts)};
for (let k = 0; k < {refresh_hz}; k++) [i, pt] = advance(ts, i, pt, 1 / {refresh_hz});
console.log(i);"""
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    assert 29 <= int(out.stdout) <= 30
