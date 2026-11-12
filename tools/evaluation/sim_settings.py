"""What the reference evaluation plays: the players, and the simulated world and robots.

`reference/players.json` names the champion and the core strategies (`evaluate.py`). `reference/world.json` is the
measured world, `reference/robots.json` the robots, assumed good. They live in a subdirectory
because `replay/fingerprint.py` hashes data files beside `match.py`: choosing a new champion must
not rerun every cached match. World and robots enter the matches together as one `SimNoise`.
"""

from __future__ import annotations

import json
from pathlib import Path

from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise

REFERENCE_DIR = Path(__file__).with_name("reference")
DEFAULT_PLAYERS = REFERENCE_DIR / "players.json"


def _values(name: str) -> dict:
    return json.loads((REFERENCE_DIR / f"{name}.json").read_text())["values"]


def load_players(path: Path = DEFAULT_PLAYERS) -> dict:
    return json.loads(Path(path).read_text())


def reference_noise(seed: int = 0) -> dict:
    """The reference world and robots as a `SimNoise` dict, played with `seed`."""
    world, robots = _values("world"), _values("robots")
    both = set(world) & set(robots)
    assert not both, f"set in both world.json and robots.json: {sorted(both)}"
    return {**world, **robots, "seed": seed}
