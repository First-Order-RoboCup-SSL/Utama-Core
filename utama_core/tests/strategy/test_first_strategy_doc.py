"""The tutorial's strategy (tutorials/first_strategy.md) still builds, so a new member's first paste works."""

import re
from pathlib import Path
from unittest.mock import MagicMock

from utama_core.engine.strategy import Strategy

DOC = Path(__file__).resolve().parents[3] / "tutorials" / "first_strategy.md"


def test_the_tutorial_strategy_builds():
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(), re.S)
    (code,) = [b for b in blocks if b.startswith("# utama_core/strategy/rookie.py")]
    module: dict = {}
    exec(compile(code, "rookie.py", "exec"), module)
    assert isinstance(module["build"]((1, 2, 3, 4, 5), MagicMock()), Strategy)
