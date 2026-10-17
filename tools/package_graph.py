"""Package dependency graph of `utama_core/`, as a Mermaid flowchart, and the stated import rules it breaks.

    pixi run python tools/package_graph.py

Parses every `.py` under `utama_core/` with `ast` (nothing is imported) and counts imports between
first-level packages (`engine`, `replay`, ...). Test code is not scanned as a source, but an import
of `utama_core.tests` from other code is an edge, to the `tests` package. Imports inside functions
and `TYPE_CHECKING` blocks count too. Self-edges are skipped, and packages with no edges are left
out of the graph. Prints:

- a Mermaid `flowchart LR` with one edge per package pair, labelled by its number of imports;
- the violations of `RULES`, each with its file and line, or `none`.

The output is pasted into `docs/architecture.md`, section "Package dependencies".
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "utama_core"
TESTS = "tests"

# (importer, imported): a package that must never import another.
RULES: tuple[tuple[str, str], ...] = (("replay", "analysis"),)


@dataclass(frozen=True)
class Import:
    src: str  # importing first-level package
    dst: str  # imported first-level package
    file: str  # path relative to the repository root
    line: int


def _absolute_target(node: ast.ImportFrom, module_parts: list[str]) -> list[str]:
    """Dotted parts a `from ... import` names, with relative levels resolved.

    `module_parts` is the importing file's path below `utama_core/`, without `.py`, so `a/b.py` gives
    [a, b]. Its package is everything but the last part (`a/__init__.py` gives [a, __init__]: same rule).
    """
    if node.level == 0:
        return (node.module or "").split(".") if node.module else []
    anchor = [PACKAGE] + module_parts[:-1]
    anchor = anchor[: len(anchor) - (node.level - 1)]
    return anchor + (node.module.split(".") if node.module else [])


def _imports_in(tree: ast.AST, module_parts: list[str], packages: set[str]) -> list[tuple[str, int]]:
    """(imported first-level package, line) for each import of `utama_core` code in one module."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE and len(parts) > 1:
                    found.append((parts[1], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            target = _absolute_target(node, module_parts)
            if target and target[0] == PACKAGE and len(target) >= 2:
                found.append((target[1], node.lineno))
            elif target == [PACKAGE]:
                # `from utama_core import engine, replay`: each name is a package.
                found.extend((alias.name, node.lineno) for alias in node.names if alias.name in packages)
    return found


def scan(root: Path = ROOT) -> list[Import]:
    """Every import between first-level packages under `root / utama_core`, skipping `tests/` sources."""
    base = root / PACKAGE
    packages = {p.name for p in base.iterdir() if p.is_dir() and p.name != "__pycache__"}
    found: list[Import] = []
    for path in sorted(base.rglob("*.py")):
        rel = path.relative_to(base)
        # Top-level modules (utama_core/__init__.py) belong to no package; tests are not sources.
        if len(rel.parts) < 2 or rel.parts[0] == TESTS:
            continue
        module_parts = list(rel.with_suffix("").parts)
        src = module_parts[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for dst, line in _imports_in(tree, module_parts, packages):
            if dst != src:
                found.append(Import(src, dst, f"{PACKAGE}/{rel.as_posix()}", line))
    return found


def edge_counts(imports: list[Import]) -> Counter[tuple[str, str]]:
    return Counter((i.src, i.dst) for i in imports)


def violations(imports: list[Import], rules: tuple[tuple[str, str], ...] = RULES) -> list[Import]:
    banned = set(rules)
    return [i for i in imports if (i.src, i.dst) in banned]


def render_mermaid(counts: Counter[tuple[str, str]]) -> str:
    """A `flowchart LR` with one labelled edge per package pair, sorted for a stable diff."""
    lines = ["flowchart LR"]
    for (src, dst), n in sorted(counts.items()):
        lines.append(f'    p_{src}["{src}"] -->|{n}| p_{dst}["{dst}"]')
    return "\n".join(lines)


def main() -> None:
    imports = scan()
    print("```mermaid")
    print(render_mermaid(edge_counts(imports)))
    print("```")
    print()
    bad = violations(imports)
    print("Violations of the import rules (`replay/` never imports `analysis/`):")
    if not bad:
        print("none")
    for i in bad:
        print(f"- {i.src} -> {i.dst}: {i.file}:{i.line}")


if __name__ == "__main__":
    main()
