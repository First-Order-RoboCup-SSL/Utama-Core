"""Pack one strategy and the code it runs on into a `.strategy` file.

    pixi run python tools/build_strategy.py split_shape split_shape.strategy
    pixi run python tools/build_strategy.py split_shape frozen.strategy --ref v2.5.0

Without `--ref` it takes the working tree as it stands, uncommitted changes included.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.bundle.strategy_file import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("strategy", help="strategy name, e.g. split_shape")
    parser.add_argument("output", type=Path)
    parser.add_argument("--ref", help="git commit to pack (default: the working tree)")
    args = parser.parse_args()
    path = build(args.strategy, args.output, args.ref)
    print(f"{path}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
