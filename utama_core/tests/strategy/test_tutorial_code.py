"""The tutorials' code still works as pasted (tutorials/*.md), so a new member's first paste runs."""

import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from utama_core.engine.strategy import Strategy

TUTORIALS = Path(__file__).resolve().parents[3] / "tutorials"


def block(page: str, first_line: str) -> str:
    """The page's one Python block that starts with `first_line`."""
    blocks = re.findall(r"```python\n(.*?)```", (TUTORIALS / page).read_text(), re.S)
    (code,) = [b for b in blocks if b.startswith(first_line)]
    return code


def run(code: str, name: str) -> dict:
    namespace: dict = {"__name__": name}
    exec(compile(code, name, "exec"), namespace)
    return namespace


def _test_functions(namespace: dict) -> list:
    return [f for name, f in namespace.items() if name.startswith("test_")]


def test_first_strategy_builds():
    module = run(block("first_strategy.md", "# utama_core/strategy/rookie.py"), "rookie")
    assert isinstance(module["build"]((1, 2, 3, 4, 5), MagicMock()), Strategy)


def test_first_tactic_builds_and_its_own_tests_pass(monkeypatch):
    strategy_code = block("first_tactic.md", "# utama_core/strategy/rookie.py")
    rookie = types.ModuleType("utama_core.strategy.rookie")
    exec(compile(strategy_code, "rookie.py", "exec"), rookie.__dict__)
    assert isinstance(rookie.build((1, 2, 3, 4, 5), MagicMock()), Strategy)

    # The page's test file imports `utama_core.strategy.rookie`: give it the page's module.
    monkeypatch.setitem(sys.modules, "utama_core.strategy.rookie", rookie)
    tests = _test_functions(run(block("first_tactic.md", "# utama_core/tests/strategy/test_rookie.py"), "test_rookie"))
    assert len(tests) == 4
    for test in tests:
        test()


def test_debug_a_moment_shows_the_open_keeper_bug():
    tests = dict(
        (f.__name__, f)
        for f in _test_functions(
            run(block("debug_a_moment.md", "# utama_core/tests/skills/test_goalkeep_retrieval.py"), "test_keeper")
        )
    )
    tests["test_a_still_ball_in_the_box_is_fetched_already"]()
    # The page says this test fails until someone fixes the keeper. Once it passes, the bug is
    # fixed: update tutorials/debug_a_moment.md (and pick another open bug), then this test.
    with pytest.raises(AssertionError):
        tests["test_the_keeper_fetches_a_ball_rolling_slowly_through_its_box"]()
