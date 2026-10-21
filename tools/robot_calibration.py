# ruff: noqa: E402
"""A robot's numbers (loop delay, speed and turn tracking, acceleration, vision noise, kick speed)
from a recorded field test. See `utama_core/analysis/robot_calibration.py`.

    pixi run python tools/robot_calibration.py replays/field/<run>.log [--json OUT]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis.robot_calibration import main

if __name__ == "__main__":
    main()
