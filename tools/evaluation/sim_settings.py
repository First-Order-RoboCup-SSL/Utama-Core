"""What the reference evaluation plays: the opponents, and the simulated world and robots.

`reference/opponents.json` is who a candidate is judged against. `reference/world.json` is the
measured world, `reference/robots.json` the robots, assumed good. They live in a subdirectory
because `replay/fingerprint.py` hashes data files beside `match.py`: choosing a new champion must
not rerun every cached match. World and robots enter the matches together as one `SimNoise`.
"""

from __future__ import annotations

import json
from pathlib import Path

from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise

REFERENCE_DIR = Path(__file__).with_name("reference")
DEFAULT_OPPONENTS = REFERENCE_DIR / "opponents.json"
_REFERENCE_SPEC = "reference"


def _values(name: str) -> dict:
    return json.loads((REFERENCE_DIR / f"{name}.json").read_text())["values"]


def load_opponents(path: Path = DEFAULT_OPPONENTS) -> dict:
    return json.loads(Path(path).read_text())


def reference_noise(seed: int = 0) -> dict:
    """The reference world and robots as a `SimNoise` dict, played with `seed`."""
    world, robots = _values("world"), _values("robots")
    both = set(world) & set(robots)
    assert not both, f"set in both world.json and robots.json: {sorted(both)}"
    return {**world, **robots, "seed": seed}


def sim_noise_from_spec(spec: str) -> SimNoise:
    """`round_robin.py --sim-noise`'s argument: `reference`, `reference:<seed>`, a JSON file or inline JSON."""
    name, _, seed = spec.partition(":")
    if name == _REFERENCE_SPEC:
        return SimNoise.from_dict(reference_noise(int(seed) if seed else 0))
    return SimNoise.from_dict(json.loads(Path(spec).read_text() if Path(spec).is_file() else spec))
