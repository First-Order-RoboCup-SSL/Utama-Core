"""`UTAMA_EXACT_MATH` selects the exact (original) arithmetic everywhere
`EXACT_MATH` switches, and anything else leaves the fast paths in place.

Checked in a fresh interpreter per value, since the switch is read once at
import and each module binds its choice then."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PROBE = """
from utama_core.config.settings import EXACT_MATH
from utama_core.tests.fixtures.exact_math import SWITCHES

picked = set()
for owner, name, exact, fast in SWITCHES:
    current = getattr(owner, name)
    assert current is exact or current is fast, (owner, name)
    picked.add("exact" if current is exact else "fast")
print(EXACT_MATH, sorted(picked))
"""


def _probe(value):
    env = {k: v for k, v in os.environ.items() if k != "UTAMA_EXACT_MATH"}
    if value is not None:
        env["UTAMA_EXACT_MATH"] = value
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=_REPO_ROOT, env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip().splitlines()[-1]


def test_exact_math_env_selects_every_exact_path():
    assert _probe("1") == "True ['exact']"


@pytest.mark.parametrize("value", [None, "0", ""])
def test_fast_paths_are_the_default(value):
    assert _probe(value) == "False ['fast']"
