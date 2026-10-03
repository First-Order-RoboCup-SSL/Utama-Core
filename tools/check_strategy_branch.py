"""check_strategy_branch.py — fail if a strategy branch could score better without playing better.

Run:
    python tools/check_strategy_branch.py --base origin/main

A strategy is scored by the round-robin and the scenario bench. If the branch that changes it, often
written by an agent, could also change the evaluation or its opponents, it could raise its score
without playing better. CI runs this on every pull request from a `strategy/*` branch, using the
copy of this file on the base branch, so the branch under test can't loosen it. Three checks:

1. **Paths.** Only `ALLOWED` changes; anything else goes on an ordinary branch, reviewed by a
   person. Renames count as a deletion and an addition (`--no-renames`).
2. **Opponents.** Every other strategy must run the base branch's code, so the round-robin plays
   the same opponents as on `main`. A branch either adds new strategy modules or changes exactly
   one existing one, never both; it never changes or deletes `pickers.py` (shared by most
   strategies) or deletes a strategy; and in `kernel_strategy.py` it only adds imports from the
   modules it adds.
3. **Reach.** A new or changed strategy module can't affect anything outside itself: no import-time
   effects or hazards (`utama_core/replay/fingerprint.py`'s `_import_time_effects` and `_hazards`:
   environment, files, dynamic imports, module-level state), no imports of another strategy
   module, and no assigning or deleting attributes of what it imports (`SomeTactic.x = ...`,
   `setattr`).

A static check, not a sandbox: the reviewer still reads the diff. Stdlib only: CI runs it
without the pixi environment.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

# A trailing "/" allows everything under that directory; anything else is one exact file.
ALLOWED = (
    "utama_core/strategy/",
    "utama_core/tests/strategy/",
    "docs/strategies.md",
)

STRATEGY_DIR = "utama_core/strategy/"
REGISTRY = STRATEGY_DIR + "kernel_strategy.py"
PICKERS = STRATEGY_DIR + "pickers.py"
_FINGERPRINT = Path("utama_core/replay/fingerprint.py")


def _allowed(path: str) -> bool:
    return any(path.startswith(a) if a.endswith("/") else path == a for a in ALLOWED)


def outside_allowlist(paths: list[str]) -> list[str]:
    """The changed paths a strategy branch may not touch, in the order given."""
    return [p for p in paths if not _allowed(p)]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def changed_files(base: str) -> dict[str, str]:
    """Changed path -> git status letter (A added, M modified, D deleted)."""
    out = _git("diff", "--name-status", "--no-renames", f"{base}...HEAD")
    return {path: status[0] for status, path in (line.split("\t", 1) for line in out.splitlines() if line)}


def changed_paths(base: str) -> list[str]:
    return sorted(changed_files(base))


def _module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def opponent_problems(changes: dict[str, str]) -> list[str]:
    """Check 2's file rules, from `changed_files`."""
    strategy = {p: s for p, s in changes.items() if p.startswith(STRATEGY_DIR) and p.endswith(".py") and p != REGISTRY}
    problems = [f"{p}: deletes a strategy module" for p, s in strategy.items() if s == "D"]
    if PICKERS in strategy:
        problems.append(f"{PICKERS}: shared by most strategies, so changing it changes the opponents")
    added = sorted(p for p, s in strategy.items() if s == "A")
    modified = sorted(p for p, s in strategy.items() if s == "M" and p != PICKERS)
    if added and modified:
        problems.append(f"adds {', '.join(added)} and also changes {', '.join(modified)}: do one or the other")
    elif len(modified) > 1:
        problems.append(f"changes {len(modified)} existing strategies ({', '.join(modified)}): change one")
    return problems


def registry_problems(old_source: str, new_source: str, added_modules: set[str]) -> list[str]:
    """Check 2's `kernel_strategy.py` rule: the old statements unchanged and in order, plus only
    `from <added module> import ...` statements."""
    old = [ast.dump(s) for s in ast.parse(old_source).body]
    new = ast.parse(new_source).body
    problems, i = [], 0
    for stmt in new:
        if i < len(old) and ast.dump(stmt) == old[i]:
            i += 1
        elif not (isinstance(stmt, ast.ImportFrom) and stmt.module in added_modules):
            problems.append(
                f"{REGISTRY} line {stmt.lineno}: only imports from the modules this branch adds may be added"
            )
    if i < len(old):
        problems.append(f"{REGISTRY}: changes or removes an existing statement")
    return problems


def _load_fingerprint(path: Path = _FINGERPRINT):
    # By path, not as `utama_core.replay.fingerprint`: the package's __init__ needs numpy, which
    # CI's plain python3 lacks. Check 1 has passed by now, so this is the base branch's file.
    spec = importlib.util.spec_from_file_location("_fingerprint", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reach_problems(path: str, source: str, fingerprint) -> list[str]:
    """Check 3 for one new or changed strategy module."""
    tree = ast.parse(source)
    problems = [f"{path}: {e}" for stmt in tree.body for e in fingerprint._import_time_effects(stmt)]
    problems += [f"{path}: {kind}" for kind in sorted(fingerprint._hazards(tree))]
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {a.asname or a.name.split(".")[0] for a in node.names}
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in modules:
                if m.startswith("utama_core.strategy") and m != _module_name(PICKERS):
                    problems.append(f"{path} line {node.lineno}: imports another strategy module ({m})")
        targets = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            base = t
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if base is not t and isinstance(base, ast.Name) and base.id in imported:
                problems.append(f"{path} line {node.lineno}: changes {ast.unparse(t)}, which it imports")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr"):
            problems.append(f"{path} line {node.lineno}: calls {node.func.id}()")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the branch the strategy branch will merge into")
    args = parser.parse_args(argv)

    changes = changed_files(args.base)
    bad = outside_allowlist(list(changes))
    if bad:
        print("A strategy branch may only change " + ", ".join(ALLOWED) + ". Outside it:")
        for path in bad:
            print(f"  {path}")
        return 1

    problems = opponent_problems(changes)
    added = {_module_name(p) for p, s in changes.items() if s == "A" and p.startswith(STRATEGY_DIR)}
    if REGISTRY in changes:
        problems += registry_problems(_git("show", f"{args.base}:{REGISTRY}"), Path(REGISTRY).read_text(), added)
    fingerprint = _load_fingerprint()
    for path, status in sorted(changes.items()):
        if status != "D" and path.startswith(STRATEGY_DIR) and path.endswith(".py") and path != REGISTRY:
            problems += reach_problems(path, Path(path).read_text(), fingerprint)
    if problems:
        print("This strategy branch could change its opponents or the evaluation:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(
        "OK: only strategy code changed, the other strategies run the base branch's code, and nothing reaches outside"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
