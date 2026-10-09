"""`tools/check_strategy_branch.py`: a strategy branch may change strategy code only, may not change
its opponents, and its code may not reach outside itself.

Lives outside `tests/strategy/` on purpose: that directory is on the allowlist, so a strategy branch
could edit a test kept there.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.check_strategy_branch import (
    ALLOWED,
    _load_fingerprint,
    changed_paths,
    main,
    opponent_problems,
    outside_allowlist,
    reach_problems,
    registry_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_strategy_modules_tests_and_catalog_are_allowed():
    assert (
        outside_allowlist(
            [
                "utama_core/strategy/new_idea.py",
                "utama_core/strategy/kernel_strategy.py",
                "utama_core/tests/strategy/test_new_idea.py",
                "docs/strategies.md",
            ]
        )
        == []
    )


@pytest.mark.parametrize(
    "path",
    [
        "utama_core/engine/strategy.py",
        "utama_core/scenario_bench/scenario_scorer.py",
        "tools/tournament/tournament_lib.py",
        "tools/check_strategy_branch.py",
        ".github/workflows/ci.yml",
        # Prefix matches on whole directory names only.
        "utama_core/strategy_runner/x.py",
        "utama_core/tests/strategy_runner/test_x.py",
        "docs/strategies.md.bak",
    ],
)
def test_anything_else_is_reported(path):
    assert outside_allowlist(["utama_core/strategy/ok.py", path]) == [path]


def test_allowlist_is_strategy_code_only():
    # Widening this is a decision, not a fix: change the test with it.
    assert ALLOWED == (
        "utama_core/strategy/",
        "utama_core/tactics/",
        "utama_core/tests/strategy/",
        "docs/strategies.md",
    )


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True)


REGISTRY_SRC = (
    '"""Factories."""\n'
    "from utama_core.strategy.a import build_a_kernel_strategy\n"
    "from utama_core.strategy.b import build_b_kernel_strategy\n"
)
STRATEGY_SRC = (
    "from utama_core.strategy.pickers import helper\n\n\ndef build_{n}_kernel_strategy(ids):\n    return helper(ids)\n"
)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "main")
    s = tmp_path / "utama_core/strategy"
    s.mkdir(parents=True)
    (s / "kernel_strategy.py").write_text(REGISTRY_SRC)
    (s / "pickers.py").write_text("def helper(ids):\n    return ids\n")
    for n in "ab":
        (s / f"{n}.py").write_text(STRATEGY_SRC.format(n=n))
    t = tmp_path / "utama_core/tactics"
    t.mkdir()
    (t / "press.py").write_text("class PressTactic:\n    reach = 1\n")
    fp = tmp_path / "utama_core/replay/fingerprint.py"
    fp.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "utama_core/replay/fingerprint.py", fp)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "strategy/x")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commit(repo, files):
    for rel, text in files.items():
        (repo / rel).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "change")


def test_tuning_one_strategy_passes(repo):
    _commit(repo, {"utama_core/strategy/a.py": STRATEGY_SRC.format(n="a").replace("helper(ids)", "helper(ids)[:3]")})
    assert main(["--base", "main"]) == 0


def test_adding_a_strategy_and_its_import_passes(repo):
    registry = REGISTRY_SRC + "from utama_core.strategy.c import build_c_kernel_strategy\n"
    _commit(
        repo,
        {"utama_core/strategy/c.py": STRATEGY_SRC.format(n="c"), "utama_core/strategy/kernel_strategy.py": registry},
    )
    assert main(["--base", "main"]) == 0


def test_adding_a_strategy_and_weakening_another_fails(repo):
    registry = REGISTRY_SRC + "from utama_core.strategy.c import build_c_kernel_strategy\n"
    _commit(
        repo,
        {
            "utama_core/strategy/c.py": STRATEGY_SRC.format(n="c"),
            "utama_core/strategy/kernel_strategy.py": registry,
            "utama_core/strategy/b.py": STRATEGY_SRC.format(n="b").replace("helper(ids)", "[]"),
        },
    )
    assert main(["--base", "main"]) == 1


def test_adding_a_tactic_and_a_strategy_that_uses_it_passes(repo):
    tactic = "from utama_core.tactics.press import PressTactic\n\n\nclass FastPress(PressTactic):\n    reach = 2\n"
    registry = REGISTRY_SRC + "from utama_core.strategy.c import build_c_kernel_strategy\n"
    strategy = "from utama_core.tactics.fast_press import FastPress\n\n\ndef build_c_kernel_strategy(ids):\n    return FastPress()\n"
    _commit(
        repo,
        {
            "utama_core/tactics/fast_press.py": tactic,
            "utama_core/strategy/c.py": strategy,
            "utama_core/strategy/kernel_strategy.py": registry,
        },
    )
    assert main(["--base", "main"]) == 0


def test_changing_an_existing_tactic_fails(repo):
    # The opponents run the existing tactics: changing one changes who you are scored against.
    _commit(repo, {"utama_core/tactics/press.py": "class PressTactic:\n    reach = 0\n"})
    assert main(["--base", "main"]) == 1


def test_a_new_tactic_may_not_reach_outside_itself(repo):
    sneaky = "from utama_core.tactics.press import PressTactic\n\nPressTactic.reach = 0\n"
    _commit(repo, {"utama_core/tactics/sneaky.py": sneaky})
    assert main(["--base", "main"]) == 1


def test_moving_a_file_out_of_strategy_fails(repo):
    (repo / "utama_core/engine").mkdir(parents=True)
    _git(repo, "mv", "utama_core/strategy/a.py", "utama_core/engine/a.py")
    _git(repo, "commit", "-qm", "move")
    assert changed_paths("main") == ["utama_core/engine/a.py", "utama_core/strategy/a.py"]
    assert main(["--base", "main"]) == 1


# --- opponents --------------------------------------------------------------------------------


def test_replacing_two_strategies_with_symlinks_fails(repo):
    # git reports a file turned into a symlink as T, not M; each still changes an opponent
    for n in "ab":
        (repo / f"utama_core/strategy/{n}.py").unlink()
        (repo / f"utama_core/strategy/{n}.py").symlink_to("pickers.py")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "symlinks")
    assert main(["--base", "main"]) == 1


def test_a_type_change_counts_as_changing_a_strategy():
    problems = opponent_problems({"utama_core/strategy/a.py": "T"})
    assert any("type" in p for p in problems)
    assert opponent_problems({"utama_core/strategy/a.py": "M", "utama_core/strategy/b.py": "T"})


def test_changing_the_strategy_package_initializer_is_reported():
    # every strategy runs it, so it is not the one strategy a branch may change
    assert opponent_problems({"utama_core/strategy/__init__.py": "M"})


def test_changing_two_existing_strategies_is_reported():
    assert opponent_problems({"utama_core/strategy/a.py": "M", "utama_core/strategy/b.py": "M"})


def test_changing_pickers_or_deleting_a_strategy_is_reported():
    assert opponent_problems({"utama_core/strategy/pickers.py": "M"})
    assert opponent_problems({"utama_core/strategy/b.py": "D"})


def test_tests_and_the_catalog_do_not_count_as_strategies():
    changes = {
        "utama_core/strategy/a.py": "M",
        "utama_core/tests/strategy/test_a.py": "A",
        "docs/strategies.md": "M",
    }
    assert opponent_problems(changes) == []


def test_registry_may_only_gain_imports_from_added_modules():
    added = {"utama_core.strategy.c"}
    new_import = "from utama_core.strategy.c import build_c_kernel_strategy\n"
    # isort may put the new import anywhere.
    first = REGISTRY_SRC.replace('"""Factories."""\n', '"""Factories."""\n' + new_import)
    assert registry_problems(REGISTRY_SRC, first, added) == []
    assert registry_problems(REGISTRY_SRC, REGISTRY_SRC + new_import, set())
    repointed = REGISTRY_SRC.replace("strategy.b import", "strategy.c import")
    assert registry_problems(REGISTRY_SRC, repointed, added)
    assert registry_problems(REGISTRY_SRC, REGISTRY_SRC + "build_b_kernel_strategy = None\n", added)


def test_an_added_module_may_not_rebind_or_rename_a_registry_name():
    added = {"utama_core.strategy.c"}
    # placed after b's import, this would replace the opponent b's factory with c's code
    shadow = REGISTRY_SRC + "from utama_core.strategy.c import build_b_kernel_strategy\n"
    assert registry_problems(REGISTRY_SRC, shadow, added)
    renamed = REGISTRY_SRC + "from utama_core.strategy.c import build_c_kernel_strategy as build_a_kernel_strategy\n"
    assert registry_problems(REGISTRY_SRC, renamed, added)


# --- reach ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fingerprint():
    return _load_fingerprint(REPO_ROOT / "utama_core/replay/fingerprint.py")


def test_a_plain_strategy_reaches_nothing(fingerprint):
    assert reach_problems("s.py", STRATEGY_SRC.format(n="c"), fingerprint) == []


@pytest.mark.parametrize(
    "body",
    [
        # Changing a shared tactic both teams use, at build time.
        "from utama_core.tactics.press import PressTactic\n\n\ndef build(ids):\n    PressTactic.reach = 0\n",
        "from utama_core.tactics import press\n\n\ndef build(ids):\n    press.TABLE['k'] = 0\n",
        "from utama_core.tactics import press\n\n\ndef build(ids):\n    setattr(press, 'reach', 0)\n",
        # Reaching another strategy, the opponent.
        "from utama_core.strategy.tiki_taka import build_tiki_taka_kernel_strategy\n",
        "import utama_core.strategy.tiki_taka\n",
        # Import-time effects and hazards.
        "import sys\n\nsys.path.append('x')\n",
        "import os\n\n\ndef build(ids):\n    return os.environ['X']\n",
        "_streak = 0\n\n\ndef build(ids):\n    global _streak\n    _streak += 1\n",
        # Through an alias of what it imports.
        "from utama_core.tactics.press import PressTactic\n\n\ndef build(ids):\n    target = PressTactic\n"
        "    target.reach = 0\n",
        "from utama_core.tactics.press import PressTactic\n\n\ndef build(ids):\n    t, n = PressTactic, 1\n"
        "    t.reach = n\n",
        # Calling a mutating method on what it imports.
        "from utama_core.tactics import press\n\n\ndef build(ids):\n    press.TABLE.clear()\n",
        # Class-level state of its own module, which both teams share.
        "class State:\n    streak = 0\n\n\ndef build(ids):\n    State.streak += 1\n",
        # Copilot on #141: through a classmethod, a wildcard import, a default, an in-place sort.
        "class State:\n    streak = 0\n\n    @classmethod\n    def bump(cls):\n        cls.streak += 1\n",
        "class State:\n    streak = 0\n\n    def bump(cls):\n        cls.streak += 1\n",
        "from utama_core.tactics.press import *\n\n\ndef build(ids):\n    PressTactic.reach = 0\n",
        "from utama_core.tactics.press import PressTactic\n\n\ndef build(ids, t=PressTactic):\n    t.reach = 0\n",
        "from utama_core.tactics.press import PressTactic\n\n\ndef build(ids, *, t=PressTactic):\n    t.reach = 0\n",
        "from utama_core.tactics import press\n\n\ndef build(ids):\n    press.TABLE.sort()\n",
        "from utama_core.tactics import press\n\n\ndef build(ids):\n    press.QUEUE.rotate(1)\n",
    ],
)
def test_reaching_outside_the_module_is_reported(fingerprint, body):
    assert reach_problems("s.py", body, fingerprint)


def test_setting_attributes_on_its_own_objects_is_fine(fingerprint):
    body = "class State:\n    pass\n\n\ndef build(ids):\n    s = State()\n    s.last = ids\n    return s\n"
    assert reach_problems("s.py", body, fingerprint) == []
    made = (
        "from utama_core.tactics.press import PressTactic\n\n\ndef build(ids):\n    t = PressTactic()\n"
        "    t.reach = 0\n    t.seen.append(ids)\n    return t\n"
    )
    assert reach_problems("s.py", made, fingerprint) == []
    own = "class State:\n    def __init__(self):\n        self.streak = 0\n\n    def bump(self):\n        self.streak += 1\n"
    assert reach_problems("s.py", own, fingerprint) == []
    local = "def build(ids, order=None):\n    xs = list(ids)\n    xs.sort()\n    return xs\n"
    assert reach_problems("s.py", local, fingerprint) == []


@pytest.mark.parametrize(
    "body", ["from .tiki_taka import build_tiki_taka_kernel_strategy\n", "from . import tiki_taka\n"]
)
def test_a_relative_import_of_another_strategy_is_reported(fingerprint, body):
    assert reach_problems("utama_core/strategy/c.py", body, fingerprint)
    assert reach_problems("utama_core/strategy/c.py", "from .pickers import helper\n", fingerprint) == []


def test_every_existing_strategy_passes_the_reach_check(fingerprint):
    # overload_flow kept a `global` possession streak, which both teams shared; it is now in
    # its picker's closure, so a strategy branch can change any existing strategy.
    failing = {
        p.name
        for p in (REPO_ROOT / "utama_core/strategy").glob("*.py")
        if p.name not in ("__init__.py", "kernel_strategy.py", "pickers.py")
        and reach_problems(p.name, p.read_text(), fingerprint)
    }
    assert failing == set()
