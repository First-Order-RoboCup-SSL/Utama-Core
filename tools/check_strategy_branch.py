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
   strategies) or deletes a strategy; in `kernel_strategy.py` it only adds imports from the
   modules it adds; and in `tactics/` it only adds new modules, since the existing tactics are
   what the opponents run (to improve one, copy it into a new module).
3. **Reach.** A new or changed strategy or tactic module can't affect anything outside itself: no import-time
   effects or hazards (`utama_core/replay/fingerprint.py`'s `_import_time_effects` and `_hazards`:
   environment, files, dynamic imports, module-level state), no imports of another strategy
   module (relative ones included), no changing what it imports or an alias of it
   (`SomeTactic.x = ...`, `t = SomeTactic; t.x = ...`, `mod.TABLE.clear()`, `setattr`), and
   no changing its own module-level classes or variables from inside a function
   (`State.streak += 1`): both teams share them.

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
    "utama_core/tactics/",  # new modules only (check 2)
    "utama_core/tests/strategy/",
    "docs/strategies.md",
)

STRATEGY_DIR = "utama_core/strategy/"
TACTICS_DIR = "utama_core/tactics/"
REGISTRY = STRATEGY_DIR + "kernel_strategy.py"
PICKERS = STRATEGY_DIR + "pickers.py"
PACKAGE_INIT = STRATEGY_DIR + "__init__.py"
_FINGERPRINT = Path("utama_core/replay/fingerprint.py")


def _allowed(path: str) -> bool:
    return any(path.startswith(a) if a.endswith("/") else path == a for a in ALLOWED)


def outside_allowlist(paths: list[str]) -> list[str]:
    """The changed paths a strategy branch may not touch, in the order given."""
    return [p for p in paths if not _allowed(p)]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def changed_files(base: str) -> dict[str, str]:
    """Changed path -> git status letter (A added, M modified, D deleted, T type changed, e.g.
    to a symlink)."""
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
    problems += [
        f"{p}: changes an existing tactic, which the opponents run; copy it into a new module instead"
        for p, s in changes.items()
        if p.startswith(TACTICS_DIR) and s != "A"
    ]
    if PICKERS in strategy:
        problems.append(f"{PICKERS}: shared by most strategies, so changing it changes the opponents")
    if PACKAGE_INIT in strategy:
        problems.append(f"{PACKAGE_INIT}: every strategy runs it, so changing it changes the opponents")
    problems += [
        f"{p}: changes the file's type (status {s}), e.g. to a symlink" for p, s in changes.items() if s == "T"
    ]
    added = sorted(p for p, s in strategy.items() if s == "A")
    # anything but an addition or a deletion changes an existing strategy, a type change included
    modified = sorted(p for p, s in strategy.items() if s not in ("A", "D") and p not in (PICKERS, PACKAGE_INIT))
    if added and modified:
        problems.append(f"adds {', '.join(added)} and also changes {', '.join(modified)}: do one or the other")
    elif len(modified) > 1:
        problems.append(f"changes {len(modified)} existing strategies ({', '.join(modified)}): change one")
    return problems


def registry_problems(old_source: str, new_source: str, added_modules: set[str]) -> list[str]:
    """Check 2's `kernel_strategy.py` rule: the old statements unchanged and in order, plus only
    `from <added module> import ...` statements, which may not rebind a name the registry
    already has (that would replace an opponent's factory) or rename what they import."""
    old_tree = ast.parse(old_source)
    old = [ast.dump(s) for s in old_tree.body]
    bound = {n.id for n in ast.walk(old_tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {
        a.asname or a.name for n in ast.walk(old_tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names
    }
    new = ast.parse(new_source).body
    problems, i = [], 0
    for stmt in new:
        if i < len(old) and ast.dump(stmt) == old[i]:
            i += 1
        elif not (isinstance(stmt, ast.ImportFrom) and stmt.module in added_modules):
            problems.append(
                f"{REGISTRY} line {stmt.lineno}: only imports from the modules this branch adds may be added"
            )
        else:
            for a in stmt.names:
                if a.asname or a.name in bound:
                    problems.append(
                        f"{REGISTRY} line {stmt.lineno}: imports {a.name}"
                        + (f" as {a.asname}" if a.asname else "")
                        + ", which would rebind or rename a name the registry has"
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


def _imported_modules(path: str, node: ast.Import | ast.ImportFrom) -> list[str]:
    """The absolute modules an import statement in `path` names, relative imports resolved
    (`from . import x` names `<package>.x`)."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if not node.level:
        return [node.module or ""]
    package = _module_name(path).split(".")[: -node.level]
    if node.module:
        return [".".join(package + [node.module])]
    return [".".join(package + [a.name]) for a in node.names]


def _root(node: ast.AST) -> ast.AST:
    """`x` of `x.a[0].b`."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node


def _aliases(tree: ast.Module, names: set[str]) -> set[str]:
    """`names` plus every name bound by a plain assignment from one of them (`t = Tactic`,
    `a, b = X, Y`, `cfg = mod.TABLE`) or as a parameter's default (`def f(t=Tactic)`), to a
    fixed point. A call (`t = Tactic()`) makes a new object, not an alias."""
    names = set(names)
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            positional = a.posonlyargs + a.args
            pairs += zip([ast.Name(id=x.arg) for x in positional[len(positional) - len(a.defaults) :]], a.defaults)
            pairs += [(ast.Name(id=x.arg), d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    pairs += zip(target.elts, node.value.elts)
                elif isinstance(target, ast.Tuple):
                    pairs += [(t, node.value) for t in target.elts]
                else:
                    pairs.append((target, node.value))
    changed = True
    while changed:
        changed = False
        for target, value in pairs:
            root = _root(value)
            if isinstance(target, ast.Name) and isinstance(root, ast.Name) and root.id in names:
                if target.id not in names:
                    names.add(target.id)
                    changed = True
    return names


def _module_bindings(tree: ast.Module) -> set[str]:
    """Names the module itself binds at its top level, classes and variables (not imports or
    functions): state every team that imports the module shares."""
    names = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


# In-place methods of the builtin containers and deque, on top of the fingerprint's list:
# called on what a module imports or shares, they change it for both teams.
_IN_PLACE = frozenset(
    {
        "sort",
        "reverse",
        "appendleft",
        "extendleft",
        "popleft",
        "rotate",
        "difference_update",
        "intersection_update",
        "symmetric_difference_update",
        "move_to_end",
        "__delitem__",
        "__iadd__",
        "__ior__",
    }
)


def _class_receivers(tree: ast.Module, shared: set[str]) -> dict[int, str]:
    """`id(node) -> receiver name` for every node in a method of a module-level class in
    `shared` whose first parameter is the class itself: a `@classmethod`, or one that calls
    it `cls`. A store through it (`cls.streak += 1`) changes state both teams share; one
    through `self` changes only that object."""
    receivers = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in shared:
            continue
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.args.args:
                continue
            first = fn.args.args[0].arg
            is_classmethod = any(isinstance(d, ast.Name) and d.id == "classmethod" for d in fn.decorator_list)
            if is_classmethod or first == "cls":
                receivers.update({id(n): first for n in ast.walk(fn)})
    return receivers


def reach_problems(path: str, source: str, fingerprint) -> list[str]:
    """Check 3 for one new or changed strategy or tactic module."""
    tree = ast.parse(source)
    problems = [f"{path}: {e}" for stmt in tree.body for e in fingerprint._import_time_effects(stmt)]
    problems += [f"{path}: {kind}" for kind in sorted(fingerprint._hazards(tree))]
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(a.name == "*" for a in node.names):
                problems.append(f"{path} line {node.lineno}: a wildcard import (its names can't be checked)")
            imported |= {a.asname or a.name.split(".")[0] for a in node.names}
            for m in _imported_modules(path, node):
                if m.startswith("utama_core.strategy") and m != _module_name(PICKERS):
                    problems.append(f"{path} line {node.lineno}: imports another strategy module ({m})")
                    break
    imported = _aliases(tree, imported)
    # module-level state changed from inside a function; at the top level it is set up once
    shared = _aliases(tree, _module_bindings(tree)) - imported
    receivers = _class_receivers(tree, shared)
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]
    in_function = {id(n) for f in functions for n in ast.walk(f) if n is not f}
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in fingerprint._MUTATORS | _IN_PLACE
        ):
            targets = [node.func.value]
            changes = f"calls {ast.unparse(node.func)}()"
        else:
            changes = None
        for t in targets:
            base = _root(t)
            if not isinstance(base, ast.Name) or (base is t and changes is None):
                continue
            if base.id in imported:
                problems.append(
                    f"{path} line {node.lineno}: {changes or 'changes ' + ast.unparse(t)}, which it imports"
                )
            elif (base.id in shared and id(node) in in_function) or receivers.get(id(node)) == base.id:
                problems.append(
                    f"{path} line {node.lineno}: {changes or 'changes ' + ast.unparse(t)}, module-level state "
                    "every team shares; keep it in the factory's closure"
                )
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
        code = path.startswith((STRATEGY_DIR, TACTICS_DIR)) and path.endswith(".py")
        if status != "D" and code and path != REGISTRY:
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
