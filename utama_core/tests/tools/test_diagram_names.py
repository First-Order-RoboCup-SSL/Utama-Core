"""Every name a drawn architecture picture shows (`NAMES` in docs/img/*/make_*.py) still exists in the code, so
a rename fails here instead of leaving a picture that is quietly wrong."""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = sorted(p for p in (REPO / "docs" / "img").glob("*/make_*.py") if "NAMES" in p.read_text())
SOURCE = "\n".join(p.read_text() for p in (REPO / "utama_core").rglob("*.py") if "tests" not in p.parts)


def names_of(script: Path) -> list[str]:
    for node in ast.parse(script.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NAMES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{script} has no module-level NAMES")


def exists(name: str) -> bool:
    """A path from the repo root, or a class, function or module-level constant defined in utama_core."""
    if "/" in name:
        return (REPO / name).exists()
    return re.search(rf"^\s*(class|def)\s+{re.escape(name)}\b|^{re.escape(name)}\s*[:=]", SOURCE, re.M) is not None


def test_there_are_drawn_pictures_to_check():
    assert any(s.name == "make_one_tick.py" for s in SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_every_name_a_picture_shows_exists(script):
    missing = [n for n in names_of(script) if not exists(n)]
    assert not missing, f"{script.relative_to(REPO)} shows names that no longer exist: {missing}"


def test_a_renamed_class_or_moved_path_is_caught():
    assert exists("PositionRefiner") and exists("utama_core/engine/strategy.py")
    assert not exists("PositionRefinerOld") and not exists("utama_core/engine/no_such_module.py")
