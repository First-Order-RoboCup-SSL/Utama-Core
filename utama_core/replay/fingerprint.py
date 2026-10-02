"""Fingerprints of the code a match runs, so a deterministic rsim result can be reused.

An rsim match is a pure function of the code each side runs, the pairing and the run
settings (`docs/STRATEGY_DEVELOPMENT.md`, "Determinism"). This module names that code:

- `strategy_fingerprint(config)`: one `build_*_kernel_strategy` factory. Inside
  `kernel_strategy.py` only the factory and the module-level names it reaches count
  (helpers, constants, imported tactic classes), so editing one factory leaves the other
  configs' fingerprints alone. Each imported repo module counts as a whole file, with
  every repo module it imports in turn (function-level imports included).
- `base_fingerprint(entry)`: everything both sides share, from the module that plays the
  match (`tools.tournament.tournament_lib` for round-robins, `utama_core.scenario_bench.scenario_scorer` for the
  bench): runner, planner, referee, sim wrapper, the rsim subprocess script, the files
  next to that code (referee profiles), both pixi environments (every installed package's
  name, version and build), the installed robosim binary, the CPU model and the
  environment variables that change numerics.
- `match_key(...)` / `bench_key(...)`: those fingerprints plus every run setting.

`kernel_strategy.py`'s own imports run in every match whichever config is playing, so a
module only one config uses still runs its import-time code in every match. A module whose
import-time code can change something outside itself (`_import_time_effects`) is moved
into the base. `audit()` lists every place the code reaches outside the import graph
(environment, files, subprocesses, dynamic imports, module-level state);
`tests/replay/test_fingerprint.py` pins that list, so a new one fails a test until someone
decides how it enters the fingerprint.

Standard library only. Nothing here imports the code it fingerprints, except
`kernel_strategy`'s file and the repo's own `.py` files, which are parsed, not run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_STRATEGY = "utama_core.strategy.kernel_strategy"
ROUND_ROBIN_ENTRY = "tools.tournament.tournament_lib"
BENCH_ENTRY = "utama_core.scenario_bench.scenario_scorer"
# `scenario_bench._runs` plays a start through these two; the bench CLI itself only reports.
BENCH_ENTRIES = (BENCH_ENTRY, "utama_core.scenario_bench.start")

# Code that runs but is not imported: `robosim_wrapper.py` starts it as a script in the
# `robosim` pixi environment.
SUBPROCESS_SCRIPTS = ("utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_subprocess",)

# Environment variables that change play: UTAMA_EXACT_MATH (`config/settings.py`) and the
# thread/JIT knobs of numba and the BLAS numpy links against.
ENV_PREFIXES = ("UTAMA_", "NUMBA_", "OMP_", "OPENBLAS_", "MKL_", "PYTHONHASHSEED")

# Files next to a module that are not data it reads.
_NOT_DATA_SUFFIXES = (".py", ".pyc", ".md")

# Method names that mutate their receiver; called at import time, they reach outside the module.
_MUTATORS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "update",
        "add",
        "setdefault",
        "pop",
        "popitem",
        "remove",
        "discard",
        "clear",
        "register",
        "__setitem__",
    }
)
_MUTATING_BUILTINS = frozenset({"setattr", "delattr", "exec", "eval", "__import__"})


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(parts: Iterable[str]) -> str:
    return _sha("\n".join(parts).encode())


@dataclass
class _Module:
    name: str
    path: Optional[Path]  # None for a namespace package (a directory without __init__.py)
    is_package: bool
    tree: Optional[ast.Module] = None
    imports: Optional[set[str]] = None  # repo modules, parents included; filled on first use

    @cached_property
    def source(self) -> bytes:
        return self.path.read_bytes() if self.path is not None else b""


class CodeGraph:
    """The repo's import graph, parsed from source (nothing is imported). A snapshot: each
    file is read once, so build a new one after editing code."""

    def __init__(self, root: Path = REPO_ROOT):
        self.root = Path(root)
        self._modules: dict[str, Optional[_Module]] = {}
        self._fingerprints: dict[tuple, str] = {}

    # --- modules -----------------------------------------------------------------------

    def module(self, name: str) -> Optional[_Module]:
        """The repo module `name`, or None if it is not in the repo (stdlib, third party)."""
        if name not in self._modules:
            self._modules[name] = self._load(name)
        return self._modules[name]

    def _load(self, name: str) -> Optional[_Module]:
        base = self.root.joinpath(*name.split("."))
        if base.with_suffix(".py").is_file():
            mod = _Module(name, base.with_suffix(".py"), is_package=False)
        elif (base / "__init__.py").is_file():
            mod = _Module(name, base / "__init__.py", is_package=True)
        elif base.is_dir() and "." in name:  # namespace package, e.g. utama_core/skills/src
            return _Module(name, None, is_package=True)
        else:
            return None
        mod.tree = ast.parse(mod.source, filename=str(mod.path))
        return mod

    def imports(self, name: str) -> set[str]:
        """The repo modules `name` imports anywhere in its file, their parents included."""
        mod = self.module(name)
        if mod.imports is None:
            found: set[str] = set()
            for node in ast.walk(mod.tree) if mod.tree is not None else ():
                for target in self._import_targets(mod, node):
                    found.update(self._with_parents(target))
            found.discard(name)
            mod.imports = found
        return mod.imports

    def _import_targets(self, mod: _Module, node: ast.AST) -> Iterable[str]:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if self.module(alias.name) is not None:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            source = self._resolve_from(mod, node)
            if source is None or self.module(source) is None:
                return
            yield source
            for alias in node.names:  # `from pkg import submodule`
                if self.module(f"{source}.{alias.name}") is not None:
                    yield f"{source}.{alias.name}"

    @staticmethod
    def _resolve_from(mod: _Module, node: ast.ImportFrom) -> Optional[str]:
        if node.level == 0:
            return node.module
        package = mod.name.split(".") if mod.is_package else mod.name.split(".")[:-1]
        if node.level - 1 > len(package):
            return None
        package = package[: len(package) - (node.level - 1)]
        return ".".join(package + ([node.module] if node.module else []))

    def _with_parents(self, name: str) -> Iterable[str]:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            if self.module(".".join(parts[:i])) is not None:
                yield ".".join(parts[:i])

    def closure(self, names: Iterable[str], cut: Iterable[str] = ()) -> set[str]:
        """Every repo module importing `names` runs, parents included; `cut` modules are
        kept but not followed."""
        cut = set(cut)
        seen: set[str] = set()
        stack = [p for n in names for p in self._with_parents(n)]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            if name not in cut:
                stack.extend(self.imports(name) - seen)
        return seen

    def file_digests(self, names: Iterable[str]) -> list[str]:
        """`relative path  sha256` for each module with a file, sorted."""
        rows = []
        for name in names:
            mod = self.module(name)
            if mod is not None and mod.path is not None:
                rows.append(f"{mod.path.relative_to(self.root).as_posix()}  {_sha(mod.source)}")
        return sorted(rows)

    def data_files(self, names: Iterable[str]) -> list[str]:
        """`relative path  sha256` of each non-code file in a module's directory (e.g. the
        referee profiles next to `profile_loader.py`), sorted."""
        dirs = set()
        for name in names:
            mod = self.module(name)
            # Not the repo root (a module there, like `conftest.py`): its neighbours are the lock file, configs, and
            # in a worktree a per-checkout `.git` file, none of which the code reads.
            if mod is not None and mod.path is not None and mod.path.parent != self.root:
                dirs.add(mod.path.parent)
        rows = []
        for d in dirs:
            for p in d.iterdir():
                if p.is_file() and not p.name.endswith(_NOT_DATA_SUFFIXES) and not p.name.startswith("."):
                    rows.append(f"{p.relative_to(self.root).as_posix()}  {_sha(p.read_bytes())}")
        return sorted(rows)

    # --- kernel_strategy.py, sliced per factory ----------------------------------------

    @cached_property
    def _kernel_index(self):
        """kernel_strategy's top level: name -> statements binding it, name -> modules an
        import binding it pulls in, and the statements every slice includes."""
        mod = self.module(KERNEL_STRATEGY)
        binders: dict[str, list[ast.stmt]] = {}
        imported: dict[str, set[str]] = {}
        always: list[ast.stmt] = []
        for stmt in mod.tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                targets = set(self._import_targets(mod, stmt))
                if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                    always.append(stmt)  # changes how the whole file compiles
                if any(alias.name == "*" for alias in stmt.names):
                    always.append(stmt)  # binds names this index can't see
                for alias in stmt.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    imported.setdefault(bound, set()).update(targets)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                binders.setdefault(stmt.name, []).append(stmt)
                if _import_time_effects(stmt):
                    always.append(stmt)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
                if _import_time_effects(stmt) or not names:
                    always.append(stmt)
                for n in names:
                    binders.setdefault(n, []).append(stmt)
            elif (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)) or _is_main_guard(stmt):
                continue  # a docstring, or code that never runs on import
            else:  # if/try/with/a bare call at module level: assume it matters to everyone
                always.append(stmt)
        return mod, binders, imported, always

    def factory_slice(self, config: str) -> tuple[list[str], set[str]]:
        """(the source of every kernel_strategy statement `config` reaches, sorted; the
        repo modules its imported names come from)."""
        mod, binders, imported, always = self._kernel_index
        if config not in binders:
            raise KeyError(f"{config} is not defined in {KERNEL_STRATEGY}")
        source = mod.source.decode()
        included: dict[int, ast.stmt] = {}
        modules: set[str] = set()
        seen_names: set[str] = set()
        stack: list[ast.stmt] = list(always) + binders[config]
        while stack:
            stmt = stack.pop()
            if id(stmt) in included:
                continue
            included[id(stmt)] = stmt
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                modules |= set(self._import_targets(mod, stmt))
            for node in ast.walk(stmt):
                names = []
                if isinstance(node, ast.Name):
                    names = [node.id]
                elif isinstance(node, (ast.Global, ast.Nonlocal)):
                    names = list(node.names)
                for n in names:
                    if n in seen_names:
                        continue
                    seen_names.add(n)
                    stack.extend(binders.get(n, []))
                    modules |= imported.get(n, set())
        return sorted(ast.get_source_segment(source, s) for s in included.values()), modules

    # --- what both sides share ----------------------------------------------------------

    def kernel_import_closure(self) -> set[str]:
        """Every repo module `import kernel_strategy` runs, whichever config plays."""
        return self.closure([KERNEL_STRATEGY]) - {KERNEL_STRATEGY}

    def ambient_modules(self) -> dict[str, list[str]]:
        """Modules kernel_strategy imports whose import-time code can reach outside them
        (module name -> what it does). They run in every match, so they join the base."""
        found = {}
        for name in sorted(self.kernel_import_closure()):
            mod = self.module(name)
            if mod.tree is None:
                continue
            effects = [e for stmt in mod.tree.body for e in _import_time_effects(stmt)]
            if effects:
                found[name] = effects
        return found

    def base_modules(self, entries: Iterable[str]) -> set[str]:
        entries = list(entries)
        cut = {KERNEL_STRATEGY}
        mods = self.closure(entries + list(SUBPROCESS_SCRIPTS), cut=cut) - cut
        # The entries only look factories up by name. Shared code that imports kernel_strategy
        # could call any helper in it, so all of the file is then shared.
        if any(KERNEL_STRATEGY in self.imports(m) for m in mods - set(entries)):
            mods.add(KERNEL_STRATEGY)
        return mods | self.closure(self.ambient_modules())

    def strategy_modules(self, config: str) -> set[str]:
        _, modules = self.factory_slice(config)
        return self.closure(modules)

    def strategy_fingerprint(self, config: str) -> str:
        key = ("strategy", config)
        if key not in self._fingerprints:
            segments, _ = self.factory_slice(config)
            mods = self.strategy_modules(config)
            self._fingerprints[key] = _digest(
                ["slice"] + [_sha(s.encode()) for s in segments] + ["files"] + self.file_digests(mods)
            )
        return self._fingerprints[key]

    def base_fingerprint(self, entries: Iterable[str]) -> str:
        key = ("base", tuple(entries))
        if key not in self._fingerprints:
            mods = self.base_modules(key[1])
            self._fingerprints[key] = _digest(
                ["files"]
                + self.file_digests(mods)
                + ["data"]
                + self.data_files(mods)
                + ["environment"]
                + [json.dumps(environment_pin(self.root), sort_keys=True)]
            )
        return self._fingerprints[key]

    # --- hazards ------------------------------------------------------------------------

    def audit(self, names: Iterable[str]) -> dict[str, list[str]]:
        """Where the code in `names` reaches outside the import graph: kind -> modules."""
        found: dict[str, set[str]] = {}
        for name in names:
            mod = self.module(name)
            if mod is None or mod.tree is None:
                continue
            for kind in _hazards(mod.tree):
                found.setdefault(kind, set()).add(name)
        return {k: sorted(v) for k, v in sorted(found.items())}


def _import_time_effects(stmt: ast.stmt) -> list[str]:
    """What `stmt` does at import time that could change anything outside its module:
    a bare call, an assignment into an attribute or subscript, a mutating method call,
    `setattr`/`exec`. Function bodies don't run at import; decorators, defaults and class
    bodies do."""
    effects = []
    if _is_main_guard(stmt):
        return effects
    for node in _evaluated_at_import(stmt):
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            effects.append(f"line {node.lineno}: bare expression")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            if any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in targets):
                effects.append(f"line {node.lineno}: assigns into an attribute or item")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _MUTATORS:
                effects.append(f"line {node.lineno}: calls .{func.attr}()")
            elif isinstance(func, ast.Name) and func.id in _MUTATING_BUILTINS:
                effects.append(f"line {node.lineno}: calls {func.id}()")
    return effects


def _evaluated_at_import(stmt: ast.stmt) -> Iterable[ast.AST]:
    """Every node of top-level `stmt` that runs when the module is imported."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        roots = stmt.decorator_list + stmt.args.defaults + [d for d in stmt.args.kw_defaults if d is not None]
        for r in roots:
            yield from ast.walk(r)
        return
    if isinstance(stmt, ast.ClassDef):
        for r in stmt.decorator_list + stmt.bases + [k.value for k in stmt.keywords]:
            yield from ast.walk(r)
        for s in stmt.body:
            yield from _evaluated_at_import(s)
        return
    yield stmt
    for child in ast.iter_child_nodes(stmt):
        if isinstance(child, ast.stmt):
            yield from _evaluated_at_import(child)
        else:
            yield from (n for n in ast.walk(child) if not isinstance(n, ast.Lambda))


def _is_main_guard(stmt: ast.stmt) -> bool:
    """`if __name__ == "__main__":`, which never runs on import."""
    test = getattr(stmt, "test", None)
    return (
        isinstance(stmt, ast.If)
        and isinstance(test, ast.Compare)
        and _is_name(test.left, "__name__")
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


_CONTAINER_CALLS = frozenset({"dict", "list", "set", "defaultdict", "OrderedDict", "deque", "Counter"})


def _hazards(tree: ast.Module) -> set[str]:
    """The kinds of reach outside the import graph in one module (see `CodeGraph.audit`)."""
    kinds = set()
    body = [s for s in tree.body if not _is_main_guard(s)]
    imported_names = set()
    containers = set()  # module-level names bound to a mutable container
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imported_names |= {a.asname or a.name.split(".")[0] for a in stmt.names}
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            v = stmt.value
            if isinstance(v, (ast.Dict, ast.List, ast.Set)) or (
                isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in _CONTAINER_CALLS
            ):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                containers |= {t.id for t in targets if isinstance(t, ast.Name)}
    for node in (n for s in body for n in ast.walk(s)):
        if isinstance(node, ast.Attribute):
            if node.attr == "environ":
                kinds.add("environment")
            elif node.attr in ("read_text", "read_bytes", "safe_load"):
                kinds.add("opens files")
            elif node.attr == "modules" and _is_name(node.value, "sys"):
                kinds.add("dynamic import")
            elif node.attr in ("Popen", "run", "call", "check_output") and _is_name(node.value, "subprocess"):
                kinds.add("subprocess")
            elif node.attr == "load" and _is_name(node.value, "json", "np", "numpy", "yaml", "pickle"):
                kinds.add("opens files")
        elif isinstance(node, ast.Name):
            if node.id == "getenv":
                kinds.add("environment")
            elif node.id == "open" and isinstance(node.ctx, ast.Load):
                kinds.add("opens files")
            elif node.id in ("__import__", "import_module", "globals", "eval", "exec"):
                kinds.add("dynamic import")
        elif isinstance(node, ast.Global):
            kinds.add("module-level state")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and containers:
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Subscript)
                    and isinstance(inner.ctx, (ast.Store, ast.Del))
                    and _is_name(inner.value, *containers)
                ) or (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr in _MUTATORS
                    and _is_name(inner.func.value, *containers)
                ):
                    kinds.add("module-level state")
        elif isinstance(node, ast.Call) and _is_name(node.func, "getattr") and len(node.args) >= 2:
            if _is_name(node.args[0], *imported_names) and not isinstance(node.args[1], ast.Constant):
                kinds.add("getattr on a module by computed name")
    return kinds


def _is_name(node: ast.AST, *ids: str) -> bool:
    return isinstance(node, ast.Name) and node.id in ids


def _pixi_env_listing(prefix: Path) -> list[str]:
    """Every package installed in a pixi environment: conda packages by name-version-build
    (conda-meta), PyPI ones by dist-info directory name."""
    rows = sorted(p.name for p in (prefix / "conda-meta").glob("*.json"))
    rows += sorted(p.name for p in prefix.glob("lib/python*/site-packages/*.dist-info"))
    return rows


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or platform.machine()


def environment_pin(root: Path = REPO_ROOT) -> dict:
    """What outside the repo changes play: both pixi environments' packages, the robosim
    binary actually installed (rebuilt from `vendor/rSim` without a version change), the
    CPU (numba compiles `fastmath` kernels for the host), and the numerics env vars."""
    robosim_env = root / ".pixi" / "envs" / "robosim"
    robosim_files = sorted(
        p
        for p in list(robosim_env.glob("lib/python*/site-packages/robosim/*")) + list(robosim_env.glob("lib/libode*"))
        if p.is_file()
    )
    return {
        "python": sys.version,
        "default_env": _digest(_pixi_env_listing(Path(sys.prefix))),
        "robosim_env": _digest(_pixi_env_listing(robosim_env)),
        "robosim_binaries": [f"{p.relative_to(robosim_env).as_posix()}  {_sha(p.read_bytes())}" for p in robosim_files],
        "cpu": _cpu_model(),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)},
    }


def match_key(
    graph: CodeGraph,
    config_a: str,
    config_b: str,
    *,
    duration_seconds: float,
    a_is_right: bool = True,
    a_kicks_off: bool = True,
    control_scheme: str = "fpp",
    fuzz_seed: Optional[int] = None,
    fuzz_interval_s: tuple[float, float] = (25.0, 45.0),
    recorded: bool = True,
) -> str:
    """Key of one `tournament_lib.run_match` call: same key, same result. `recorded` is
    whether it ran with a `run_dir` (the match log and stats recorder are then live)."""
    settings = {
        "duration_seconds": duration_seconds,
        "a_is_right": a_is_right,
        "a_kicks_off": a_kicks_off,
        "control_scheme": control_scheme,
        "fuzz_seed": fuzz_seed,
        "fuzz_interval_s": list(fuzz_interval_s) if fuzz_seed is not None else None,
        "recorded": recorded,
    }
    return _digest(
        [
            "round-robin match",
            graph.base_fingerprint([ROUND_ROBIN_ENTRY]),
            config_a,
            graph.strategy_fingerprint(config_a),
            config_b,
            graph.strategy_fingerprint(config_b),
            json.dumps(settings, sort_keys=True),
        ]
    )


def bench_key(graph: CodeGraph, start: dict, candidate: str, opponent: str, *, horizon_s: float) -> str:
    """Key of one bench start played by `score_scenario`: `start` is the jittered
    `BenchScenario.to_dict()`, so the bank, the start and the jitter seed are all in it."""
    return _digest(
        [
            "bench start",
            graph.base_fingerprint(BENCH_ENTRIES),
            json.dumps(start, sort_keys=True),
            candidate,
            graph.strategy_fingerprint(candidate),
            opponent,
            graph.strategy_fingerprint(opponent),
            json.dumps({"horizon_s": horizon_s}),
        ]
    )


def config_names(graph: CodeGraph) -> list[str]:
    """The round-robin's configs, as `tournament_lib._CONFIG_NAMES` finds them, from source."""
    _, binders, _, _ = graph._kernel_index
    return sorted(
        n
        for n, stmts in binders.items()
        if n.startswith("build_")
        and n.endswith("_kernel_strategy")
        and n != "build_default_kernel_strategy"
        and any(isinstance(s, ast.FunctionDef) for s in stmts)
    )
