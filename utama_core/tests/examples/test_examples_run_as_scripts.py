"""Each example runs as its docstring says, `pixi run python examples/<name>.py`, from anywhere:
`utama_core` is not installed in the pixi environment, so an example finds it itself. (Until
2026-10-21 every example failed at its first import that way.)"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


@pytest.mark.parametrize("name", ["real_attacker_defender.py", "demo_referee_feedback_gui.py"])
def test_an_example_with_options_prints_its_help_run_by_path(name, tmp_path):
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / name), "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "usage:" in done.stdout
