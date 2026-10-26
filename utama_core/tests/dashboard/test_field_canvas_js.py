"""`field_canvas.js` run in node with a 2D context that records lines: a robot's heading in a
mirrored view (our goal on the right, `myTeamIsRight` false) was drawn without the mirror, so a
robot facing +x pointed the wrong way in rsim and replays. Skipped where node is missing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).parents[2] / "dashboard" / "static" / "field_canvas.js").read_text()

HARNESS = """
const lines = [];
let at = null;
const ctx = new Proxy({}, {
  get: (_, name) => name === "moveTo" ? (x, y) => (at = [x, y])
    : name === "lineTo" ? (x, y) => lines.push([at, [x, y]])
    : name === "measureText" ? () => ({ width: 0 })
    : () => {},
  set: () => true,
});
const canvas = { width: 800, height: 600, getContext: () => ctx, parentElement: { clientWidth: 800, clientHeight: 600 }, style: {} };
global.ResizeObserver = class { observe() {} };
const view = new FieldCanvas(canvas, { half_length: 2, half_width: 1.5, goal_depth: 0.18 }, { myTeamIsRight: MY_TEAM_IS_RIGHT });
const bot = { id: 1, x: 0.5, y: 0.2, orientation: 0, vx: 0, vy: 0 };
view.draw({ robots: { friendly: [bot], enemy: [] }, ball: null });
const t = view._transform();
const centre = [t.toX(bot.x), t.toY(bot.y)];
const heading = lines.find(([from]) => from && Math.abs(from[0] - centre[0]) < 1e-6 && Math.abs(from[1] - centre[1]) < 1e-6);
const ahead = [t.toX(bot.x + 0.1), t.toY(bot.y)];  // where facing +x points on screen
console.log(JSON.stringify({ centre, heading: heading && heading[1], ahead, back: t.toField(...centre) }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize("ours_on_right", [True, False])
def test_a_robot_facing_plus_x_points_to_plus_x_on_screen_mirrored_or_not(ours_on_right):
    js = SCRIPT + HARNESS.replace("MY_TEAM_IS_RIGHT", json.dumps(ours_on_right))
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    (cx, cy), (hx, hy), (ax, _) = out["centre"], out["heading"], out["ahead"]
    assert (hx - cx) * (ax - cx) > 0 and hy == pytest.approx(cy)
    assert out["back"]["x"] == pytest.approx(0.5) and out["back"]["y"] == pytest.approx(0.2)
