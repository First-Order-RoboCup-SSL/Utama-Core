"""Each example runs as its docstring says, `pixi run python examples/<name>.py`, from anywhere:
`utama_core` is not installed in the pixi environment, so an example finds it itself. (Until
2026-10-21 every example failed at its first import that way.)"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "examples"


@pytest.mark.parametrize(
    "name",
    [
        "tools/field/shooting_drill.py",
        "examples/score_goal.py",
        "examples/dribble_and_shoot.py",
        "examples/pass_receive.py",
        "examples/passing_rally.py",
        "examples/save_and_counter.py",
        "tools/field/calibrate_robot.py",
        "examples/planner_ab.py",
        "examples/demos.py",
        "examples/teleop.py",
        "examples/demo_referee_feedback_gui.py",
    ],
)
def test_an_example_with_options_prints_its_help_run_by_path(name, tmp_path):
    done = subprocess.run(
        [sys.executable, str(REPO / name), "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "usage:" in done.stdout


def test_the_field_example_holds_while_its_tactic_does_not_apply():
    """`press` applies only with an opponent near the ball; picking it anyway made the kernel
    raise, stopping the run the first time the defender played it."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        import tools.field.shooting_drill as example
    finally:
        sys.path.remove(str(EXAMPLES))
    robots = frozenset({1})
    assert example._play_or_hold(None, robots, None, frozenset({"hold"})) == {"hold": robots}
    assert example._play_or_hold(None, robots, None, frozenset({"play", "hold"})) == {"play": robots}
    assert example._play_or_hold(None, frozenset(), None, frozenset({"play", "hold"})) == {}


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_every_example_imports(path, tmp_path):
    """Every example's imports resolve. Until 2026-11-11 two demos still imported the removed
    `utama_core.strategy.kernel_strategy` (gone when strategies became `strategy/<name>.py`) and one
    the long-deleted `WanderingStrategy`, and nothing ran them."""
    code = (
        "import importlib.util, sys; "
        f"sys.path.insert(0, {str(EXAMPLES)!r}); "
        f"spec = importlib.util.spec_from_file_location('example_under_test', {str(path)!r}); "
        "module = sys.modules['example_under_test'] = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
