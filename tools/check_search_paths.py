"""check_search_paths.py — fail if a strategy-search branch changed anything outside strategy code.

Run:
    python tools/check_search_paths.py --base origin/main

A search agent is scored by the round-robin and the scenario bench. If its branch could also edit
the engine, the tactics, the referee, the simulator or the evaluation tools, it could raise its
score without playing better. So a search branch may change only `ALLOWED`; everything else is a
human change, made on an ordinary branch. CI runs this on every pull request from a `search/*`
branch, using the copy of this file on the base branch, so the branch under test can't loosen it.

Renames are listed as a deletion and an addition (`--no-renames`), so moving a file out of an
allowed directory is caught too. Stdlib only: CI runs it without the pixi environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# A trailing "/" allows everything under that directory; anything else is one exact file.
ALLOWED = (
    "utama_core/strategy/",
    "utama_core/tests/strategy/",
    "docs/strategies.md",
)


def _allowed(path: str) -> bool:
    return any(path.startswith(a) if a.endswith("/") else path == a for a in ALLOWED)


def outside_allowlist(paths: list[str]) -> list[str]:
    """The changed paths a search branch may not touch, in the order given."""
    return [p for p in paths if not _allowed(p)]


def changed_paths(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the branch the search branch will merge into")
    args = parser.parse_args(argv)

    bad = outside_allowlist(changed_paths(args.base))
    if not bad:
        print(f"OK: every change is under {', '.join(ALLOWED)}")
        return 0
    print("A search branch may only change " + ", ".join(ALLOWED) + ". Outside it:")
    for path in bad:
        print(f"  {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
