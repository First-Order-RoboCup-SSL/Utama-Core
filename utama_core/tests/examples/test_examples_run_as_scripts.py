"""Each example runs as its docstring says, `pixi run python examples/<name>.py`, from anywhere:
`utama_core` is not installed in the pixi environment, so an example finds it itself. (Until
2026-10-21 every example failed at its first import that way.)"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


@pytest.mark.parametrize(
    "name",
    [
        "shooting_drill.py",
        "score_goal.py",
        "dribble_and_shoot.py",
        "pass_receive.py",
        "passing_rally.py",
        "save_and_counter.py",
        "calibrate_robot.py",
        "planner_ab.py",
        "demos.py",
        "teleop.py",
        "demo_referee_feedback_gui.py",
    ],
)
def test_an_example_with_options_prints_its_help_run_by_path(name, tmp_path):
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / name), "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "usage:" in done.stdout


def test_the_field_example_holds_while_its_tactic_does_not_apply():
    """`press` applies only with an opponent near the ball; picking it anyway made the kernel
    raise, stopping the run the first time the defender played it."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        import shooting_drill as example
    finally:
        sys.path.remove(str(EXAMPLES))
    robots = frozenset({1})
    assert example._play_or_hold(None, robots, None, frozenset({"hold"})) == {"hold": robots}
    assert example._play_or_hold(None, robots, None, frozenset({"play", "hold"})) == {"play": robots}
    assert example._play_or_hold(None, frozenset(), None, frozenset({"play", "hold"})) == {}
