# ruff: noqa: E402
"""Run the trained 1v1 striker through the field-test runner.

    pixi run python examples/real_striker.py
    pixi run python examples/real_striker.py --mode grsim

Uses FIELD, CAMERA_HALF, ATTACKER, DEFENDER and their hardware profiles from
real_attacker_defender.py. The default field is GERF, 3 m x 2.25 m. Configure the vision and
radio IDs there before a real run. Both teams start in HALT; use the dashboard at
http://localhost:8080 to check vision/radio, Force Start and Halt. Starts are placed by hand,
and the runner keeps the existing field checks, camera boundary fence, recording and real
robot speed limits. The policy was trained in rsim; this entry point does not run evaluation
episodes or automatically release the robots after a stoppage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import real_attacker_defender as setup

from tools.striker_duel import StrikerTactic, load_params, net_chooser
from utama_core.run import StrategyRunner

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "assets/models/striker_rl2_w150.npz"


def build_runner(mode: str, weights: Path = DEFAULT_WEIGHTS) -> StrategyRunner:
    params = load_params(weights)
    attacker_id = setup.ATTACKER.vision_id if mode == "real" else 0

    def striker():
        return StrikerTactic(net_chooser(params, True, np.random.default_rng(9999)), robot_id=attacker_id)

    return setup.build_runner(mode, attacker_tactics={"striker_rl": striker})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS, help="network .npz; default: bundled striker_rl2_w150"
    )
    parser.add_argument("--mode", choices=("real", "grsim", "rsim"), default="real")
    args = parser.parse_args()
    build_runner(args.mode, args.weights).run()


if __name__ == "__main__":
    main()
