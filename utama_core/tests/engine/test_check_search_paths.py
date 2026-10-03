"""`tools/check_search_paths.py`: a search branch may change strategy code only.

Lives outside `tests/strategy/` on purpose: that directory is on the allowlist, so a search branch
could edit a test kept there.
"""

import subprocess

import pytest

from tools.check_search_paths import ALLOWED, changed_paths, main, outside_allowlist


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
        "utama_core/tactics/give_and_go.py",
        "utama_core/scenario_bench/scenario_scorer.py",
        "tools/tournament/tournament_lib.py",
        "tools/check_search_paths.py",
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
    assert ALLOWED == ("utama_core/strategy/", "utama_core/tests/strategy/", "docs/strategies.md")


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "utama_core/strategy").mkdir(parents=True)
    (tmp_path / "utama_core/strategy/a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "search/x")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_branch_touching_only_strategy_passes(repo):
    (repo / "utama_core/strategy/a.py").write_text("x = 2\n")
    _git(repo, "commit", "-qam", "tune")
    assert main(["--base", "main"]) == 0


def test_moving_a_file_out_of_strategy_fails(repo):
    (repo / "utama_core/engine").mkdir(parents=True)
    _git(repo, "mv", "utama_core/strategy/a.py", "utama_core/engine/a.py")
    _git(repo, "commit", "-qm", "move")
    assert changed_paths("main") == ["utama_core/engine/a.py", "utama_core/strategy/a.py"]
    assert main(["--base", "main"]) == 1
