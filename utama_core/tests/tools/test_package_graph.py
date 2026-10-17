"""`tools/package_graph.py` on a synthetic `utama_core/` tree: edges, relative imports, skipped tests, the rule."""

from collections import Counter
from pathlib import Path

from tools.package_graph import edge_counts, render_mermaid, scan, violations


def _write(root: Path, rel: str, text: str) -> None:
    path = root / "utama_core" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(root: Path) -> None:
    _write(
        root,
        "engine/strategy.py",
        "from utama_core.entities.game import Game\nfrom . import helpers\nfrom utama_core.tests.common import z\n",
    )
    _write(root, "engine/helpers.py", "from ..replay import x\n")  # relative: engine -> replay
    _write(root, "replay/r.py", "import utama_core.analysis.z\n")  # the rule's violation
    _write(root, "replay/__init__.py", "from utama_core import analysis\n")  # `from utama_core import pkg`
    _write(root, "analysis/z.py", "x = 1\n")
    _write(root, "entities/game.py", "from utama_core.config import c\n")
    _write(root, "tests/test_x.py", "from utama_core.run import y\n")  # test code is not a source


def test_scan_counts_package_edges_and_skips_tests(tmp_path):
    _tree(tmp_path)
    imports = scan(tmp_path)
    assert edge_counts(imports) == Counter(
        {
            ("engine", "entities"): 1,
            ("engine", "tests"): 1,
            ("engine", "replay"): 1,
            ("replay", "analysis"): 2,
            ("entities", "config"): 1,
        }
    )
    assert all(i.src != "run" for i in imports)  # the only `run` import was in tests/


def test_violation_lists_each_offending_import_with_its_location(tmp_path):
    _tree(tmp_path)
    bad = violations(scan(tmp_path))
    assert sorted((i.file, i.line) for i in bad) == [
        ("utama_core/replay/__init__.py", 1),
        ("utama_core/replay/r.py", 1),
    ]
    assert violations(scan(tmp_path), rules=(("engine", "entities"),))[0].dst == "entities"


def test_no_violation_on_a_clean_tree(tmp_path):
    _write(tmp_path, "engine/a.py", "from utama_core.replay import r\n")
    _write(tmp_path, "replay/r.py", "x = 1\n")
    assert violations(scan(tmp_path)) == []


def test_mermaid_has_one_labelled_edge_per_pair():
    text = render_mermaid(Counter({("engine", "entities"): 3, ("replay", "analysis"): 1}))
    assert text.splitlines() == [
        "flowchart LR",
        '    p_engine["engine"] -->|3| p_entities["entities"]',
        '    p_replay["replay"] -->|1| p_analysis["analysis"]',
    ]
