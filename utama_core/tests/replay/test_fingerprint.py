"""`utama_core.replay.fingerprint`: which edits change which fingerprints.

Most tests build a small repo in `tmp_path` with the real layout (one module per strategy and
a shared `pickers.py`, tactic modules, a planner the runner imports, `tools/evaluation/match.py`), edit one file, and check exactly the
fingerprints that should move do. The rest run against this repo itself."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from utama_core.replay import fingerprint as fp
from utama_core.replay import match_results
from utama_core.replay.fingerprint import MATCH_ENTRY, REPO_ROOT, CodeGraph
from utama_core.strategy import catalog

PICKERS = """\
_MARGIN = 0.05  # used by the shared helper


def _shared_helper(x):
    return x + _MARGIN
"""

STRATEGY_A = """\
from __future__ import annotations

from utama_core.strategy.pickers import _shared_helper
from utama_core.tactics.attack import AttackTactic


def a(ids):
    return [AttackTactic(), _shared_helper(1)]
"""

STRATEGY_B = """\
from __future__ import annotations

from utama_core.strategy.pickers import _shared_helper
from utama_core.tactics.defend import DefendTactic

_ONLY_B = 3
_streak = 0


def _b_picker(x):
    global _streak
    _streak += 1
    return _shared_helper(x) * _ONLY_B


def b(ids):
    return [DefendTactic(), _b_picker(2)]
"""

# imports a tactic it never uses: an import counts whether or not the strategy uses it
STRATEGY_C = """\
from __future__ import annotations

from utama_core.tactics.unused import UnusedTactic


def c(ids):
    # no helper, no tactic of its own
    return [len(ids)]
"""

FILES = {
    "tools/evaluation/__init__.py": "",
    "tools/evaluation/match.py": "from utama_core.run.runner import run\nfrom utama_core.strategy import catalog\n",
    "utama_core/__init__.py": '__version__ = "1.0.0"\n',
    "utama_core/run/__init__.py": "",
    "utama_core/run/runner.py": "from ..planning import planner\nfrom utama_core.profiles import loader\n\n\ndef run():\n"
    "    return planner.plan()\n",
    "utama_core/planning/__init__.py": "",
    "utama_core/planning/planner.py": "def plan():\n    return 1\n",
    "utama_core/profiles/__init__.py": "",
    "utama_core/profiles/loader.py": "def load():\n    return open('sim.yaml').read()\n",
    "utama_core/profiles/sim.yaml": "speed: 1\n",
    "utama_core/strategy/__init__.py": "",
    "utama_core/strategy/catalog.py": "def names():\n    return []\n",
    "utama_core/strategy/pickers.py": PICKERS,
    "utama_core/strategy/a.py": STRATEGY_A,
    "utama_core/strategy/b.py": STRATEGY_B,
    "utama_core/strategy/c.py": STRATEGY_C,
    "utama_core/tactics/__init__.py": "",
    "utama_core/tactics/attack.py": "from utama_core.skills.kick import kick\n\n\nclass AttackTactic:\n"
    "    def tick(self):\n        return kick()\n",
    # `kick` imports its helper inside the function: still in the closure
    "utama_core/skills/kick.py": "def kick():\n    from utama_core.skills.aim import aim\n\n    return aim()\n",
    "utama_core/skills/aim.py": "def aim():\n    return 0\n",
    "utama_core/tactics/defend.py": "class DefendTactic:\n    pass\n",
    "utama_core/tactics/unused.py": 'class UnusedTactic:\n    pass\n\n\nif __name__ == "__main__":\n    print(1)\n',
    "utama_core/simulator/simulators/robosim/robosim_subprocess.py": "import json\n",
}
CONFIGS = ["a", "b", "c"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for pkg in ("utama_core/simulator", "utama_core/simulator"):
        (tmp_path / pkg / "__init__.py").write_text("")
    return tmp_path


def _fingerprints(root: Path) -> dict[str, str]:
    g = CodeGraph(root)
    out = {c: g.strategy_fingerprint(c) for c in CONFIGS}
    out["base"] = g.base_fingerprint([MATCH_ENTRY])
    return out


def _changed_after(root: Path, rel: str, old: str, new: str) -> set[str]:
    before = _fingerprints(root)
    path = root / rel
    text = path.read_text()
    assert old in text, f"{old!r} not in {rel}"
    path.write_text(text.replace(old, new, 1))
    after = _fingerprints(root)
    return {k for k in before if before[k] != after[k]}


PICKERS_PY = "utama_core/strategy/pickers.py"
A_PY, B_PY, C_PY = ("utama_core/strategy/a.py", "utama_core/strategy/b.py", "utama_core/strategy/c.py")


def test_editing_one_strategy_changes_only_that_config(repo):
    assert (_changed_after(repo, C_PY, "return [len(ids)]", "return [len(ids) + 1]")) == {"c"}
    assert (_changed_after(repo, A_PY, "AttackTactic(), ", "AttackTactic(), 0, ")) == {"a"}


def test_editing_a_constant_or_state_in_one_strategy_changes_only_that_config(repo):
    assert (_changed_after(repo, B_PY, "_ONLY_B = 3", "_ONLY_B = 4")) == {"b"}
    assert (_changed_after(repo, B_PY, "_streak = 0", "_streak = 5")) == {"b"}


def test_editing_the_shared_pickers_changes_the_configs_that_import_them(repo):
    assert (_changed_after(repo, PICKERS_PY, "return x + _MARGIN", "return x - _MARGIN")) == {"a", "b"}
    assert (_changed_after(repo, PICKERS_PY, "_MARGIN = 0.05", "_MARGIN = 0.06")) == {"a", "b"}


def test_editing_a_tactic_changes_exactly_the_configs_that_import_it(repo):
    rel = "utama_core/tactics/attack.py"
    assert (_changed_after(repo, rel, "return kick()", "return kick() + 1")) == {"a"}


def test_a_function_level_import_is_followed(repo):
    assert (_changed_after(repo, "utama_core/skills/aim.py", "return 0", "return 1")) == {"a"}


def test_an_imported_tactic_counts_even_if_the_strategy_never_uses_it(repo):
    assert (_changed_after(repo, "utama_core/tactics/unused.py", "pass", "x = 1")) == {"c"}


def test_a_tactic_no_strategy_imports_changes_nothing(repo):
    (repo / "utama_core/tactics/orphan.py").write_text("class Orphan:\n    pass\n")
    assert _changed_after(repo, "utama_core/tactics/orphan.py", "pass", "x = 1") == set()


def _fixture(
    g,
    a,
    b,
    *,
    duration=65.0,
    sim_noise=None,
    a_is_right=True,
    a_kicks_off=True,
):
    """The fixture id `match_results` gives a match of `a` (config_a) against `b`."""
    env = match_results.environment(
        g,
        duration_seconds=duration,
        sim_noise=sim_noise,
    )
    seed = sim_noise.get("seed") if sim_noise else None
    return match_results.fixture_id(
        env,
        g.strategy_fingerprint(a)[:16],
        g.strategy_fingerprint(b)[:16],
        seed,
        a_is_right=a_is_right,
        a_kicks_off=a_kicks_off,
    )


def test_editing_shared_infrastructure_changes_the_base_and_every_fixture(repo):
    g = CodeGraph(repo)
    fixtures = {(a, b): _fixture(g, a, b) for a in CONFIGS for b in CONFIGS if a < b}
    changed = _changed_after(repo, "utama_core/planning/planner.py", "return 1", "return 2")
    assert changed == {"base"}
    g = CodeGraph(repo)
    assert all(_fixture(g, a, b) != k for (a, b), k in fixtures.items())


def _runner_also_imports(repo: Path, module: str) -> None:
    path = repo / "utama_core/run/runner.py"
    path.write_text(f"import {module}\n" + path.read_text())


def _add(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_import_time_effects_stay_with_the_strategy_that_imports_them(repo):
    # Only the two strategies playing are imported, so nothing else can see them.
    rel = "utama_core/tactics/unused.py"
    assert _changed_after(
        repo, rel, "class UnusedTactic:", "import sys\nsys.path.append('x')\n\n\nclass UnusedTactic:"
    ) == {"c"}


def test_code_behind_the_process_boundary_is_the_players_even_if_the_runner_imports_it(repo):
    # A skills edit must not replay matches between two `.strategy` files, which carry their own.
    _runner_also_imports(repo, "utama_core.skills.aim")
    assert _changed_after(repo, "utama_core/skills/aim.py", "return 0", "return 1") == {"a"}


def test_what_the_match_process_runs_itself_stays_in_the_base(repo):
    _add(repo, "utama_core/engine/__init__.py", "")
    _add(repo, "utama_core/engine/match_log.py", "LIMIT = 1\n")
    _runner_also_imports(repo, "utama_core.engine.match_log")
    assert _changed_after(repo, "utama_core/engine/match_log.py", "LIMIT = 1", "LIMIT = 2") == {"base"}


def test_a_compiled_planner_counts_for_the_players_that_load_it_and_not_the_base(repo):
    _add(repo, "utama_core/skills/_planner.so", "version 1")
    _runner_also_imports(repo, "utama_core.skills.aim")
    assert _changed_after(repo, "utama_core/skills/_planner.so", "version 1", "version 2") == {"a"}


def test_the_rsim_subprocess_script_and_data_files_are_in_the_base(repo):
    rel = "utama_core/simulator/simulators/robosim/robosim_subprocess.py"
    assert _changed_after(repo, rel, "import json", "import json, struct") == {"base"}
    assert _changed_after(repo, "utama_core/profiles/sim.yaml", "speed: 1", "speed: 2") == {"base"}


def test_fingerprints_do_not_depend_on_where_the_checkout_is(repo, tmp_path_factory):
    import shutil

    other = tmp_path_factory.mktemp("worktree") / "elsewhere"
    shutil.copytree(repo, other)
    (other / ".git").write_text("gitdir: /some/other/checkout\n")
    assert _fingerprints(other) == _fingerprints(repo)


def test_a_release_version_bump_changes_nothing(repo):
    # The release workflow rewrites `__version__` on every merge into main; hashing it emptied
    # the whole match cache after each merge.
    rel = "utama_core/__init__.py"
    assert _changed_after(repo, rel, '__version__ = "1.0.0"', '__version__ = "2.0.1"') == set()
    assert _changed_after(repo, rel, '__version__ = "2.0.1"', '__version__ = "2.0.1"\nDEBUG = True') == {"base"} | set(
        CONFIGS
    )


def test_a_relative_import_resolves(repo):
    g = CodeGraph(repo)
    assert "utama_core.planning.planner" in g.closure(["utama_core.run.runner"])


def test_the_fixture_covers_every_setting(repo):
    g = CodeGraph(repo)
    a, b = CONFIGS[:2]
    ref = _fixture(g, a, b)
    variants = [
        _fixture(g, b, a),
        _fixture(g, a, b, duration=20.0),
        _fixture(g, a, b, a_is_right=False),
        _fixture(g, a, b, a_kicks_off=False),
        _fixture(g, a, b, sim_noise={"seed": 1}),
        _fixture(g, a, b, sim_noise={"seed": 2}),
    ]
    assert len({ref, *variants}) == 1 + len(variants)
    # the noise seed is per match, the rest of the noise per environment
    assert _fixture(g, a, b, sim_noise={"seed": 1, "kick_speed_sd": 0.2}) != variants[4]


def test_numerics_env_vars_change_the_base(repo, monkeypatch):
    monkeypatch.delenv("UTAMA_EXACT_MATH", raising=False)
    before = CodeGraph(repo).base_fingerprint([MATCH_ENTRY])
    monkeypatch.setenv("UTAMA_EXACT_MATH", "1")
    assert CodeGraph(repo).base_fingerprint([MATCH_ENTRY]) != before
    with_exact = CodeGraph(repo).base_fingerprint([MATCH_ENTRY])
    monkeypatch.setenv("UNRELATED_VAR", "1")
    assert CodeGraph(repo).base_fingerprint([MATCH_ENTRY]) == with_exact


# --- this repo ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph() -> CodeGraph:
    return CodeGraph(REPO_ROOT)


def _players() -> list[str]:
    return catalog.names()


def test_editing_one_real_strategy_does_not_change_another_configs_fingerprint(graph):
    """The point of one module per strategy: a strategy's fingerprint reaches its own module and
    the shared pickers, never a sibling strategy's module."""
    modules = {c: f"utama_core.strategy.{c}" for c in _players()}
    for config, module in modules.items():
        assert graph.strategy_modules(config) & set(modules.values()) == {module}, config
    assert graph.strategy_fingerprint("weaver") != graph.strategy_fingerprint("riptide")


def test_every_tactic_a_strategy_builds_is_in_its_closure(graph):
    """Independent of the static analysis: build every `Strategy` and check the module of each
    tactic object it holds is in that player's fingerprinted modules."""
    for config in _players():
        strategy = catalog.builder(config, (1, 2, 3, 4, 5))(None)
        modules = {type(t).__module__ for t in strategy._tactics.values()}
        assert modules <= graph.strategy_modules(config), (config, modules - graph.strategy_modules(config))


# Imported by the runner only to reset their per-match state, so a strategy that doesn't import
# them isn't affected by them, and the one that does has them in its own closure.
_RUNNER_ONLY_RESETS = {
    "utama_core.geometry.pass_and_score_geometry",
    "utama_core.skills.shielding",
    "utama_core.strategy.pickers",
}


def test_all_code_taken_out_of_the_base_is_in_every_players_fingerprint(graph):
    """Otherwise an edit there would change a match and no fingerprint at all."""
    roots = [MATCH_ENTRY, *fp.SUBPROCESS_SCRIPTS]
    resets = graph.closure(_RUNNER_ONLY_RESETS)
    taken_out = graph.closure(roots) - graph.base_modules([MATCH_ENTRY]) - resets
    for config in _players():
        assert taken_out <= graph.strategy_modules(config), (config, taken_out - graph.strategy_modules(config))


# Every place the match process calls into `PLAYER_SIDE`. A new one fails this test: is it behind
# the boundary (then a `.strategy` file must do it in its own process), or `RUNNER_SIDE`?
_RUNNER_REACHES_INTO_PLAYER_SIDE = {
    "tools.evaluation.match": {"utama_core.strategy"},  # the catalog's names
    "utama_core.run.sim_setup": {"utama_core.engine", "utama_core.strategy"},  # builds the code-named ones (load_side)
    "utama_core.bundle.strategy_file": {"utama_core.engine"},
    "utama_core.run.side_runtime": {"utama_core.engine", "utama_core.motion_planning"},
    "utama_core.run.strategy_runner": {
        "utama_core.engine",
        "utama_core.motion_planning",  # builds a controller; a file's process builds its own
        "utama_core.geometry.pass_and_score_geometry",  # reset
        "utama_core.skills",  # reset
        "utama_core.strategy",  # reset
    },
    "utama_core.team_controller.controllers.real.real_robot_controller": {"utama_core.skills"},  # real only
    "utama_core.tests.common.abstract_test_manager": {"utama_core.engine"},  # a type
}


def test_the_match_process_reaches_into_player_side_code_only_where_listed(graph):
    found: dict[str, set[str]] = {}
    for module in graph.base_modules([MATCH_ENTRY]):
        if module in fp.RUNNER_SIDE:
            continue
        for target in graph.imports(module):
            package = next((p for p in fp.PLAYER_SIDE if target == p or target.startswith(p + ".")), None)
            if package is not None and target not in fp.RUNNER_SIDE:
                found.setdefault(module, set()).add(package)
    assert found == _RUNNER_REACHES_INTO_PLAYER_SIDE


def test_the_tactics_are_outside_the_base(graph):
    """Otherwise a tactic edit would rerun every match: the per-player split is the point."""
    base = graph.base_modules([MATCH_ENTRY])
    for name in ("give_and_go", "press_and_contain", "shadow_and_mark", "switch_of_play", "defense"):
        assert f"utama_core.tactics.{name}" not in base


def test_the_base_holds_the_sim_script_and_referee_profiles(graph):
    base = graph.base_modules([MATCH_ENTRY])
    assert set(fp.SUBPROCESS_SCRIPTS) <= base
    assert any("referee/profiles/simulation.yaml" in row for row in graph.data_files(base))


# Every place the code a match runs reaches outside the import graph, and how each is
# handled. A new entry here fails this test: decide how it enters the fingerprint (or why
# it doesn't need to), then add it.
#   environment: settings.py reads UTAMA_EXACT_MATH -> ENV_PREFIXES; robosim_wrapper copies
#     the environment into the sim subprocess -> ENV_PREFIXES covers the numerics ones.
#   opens files: profile_loader reads profiles/*.yaml -> data_files. The rest write a
#     match's outputs (match log, stats, replays, a camera run's game log) or read old ones for
#     analysis, never during play; catalog reads strategy sources to list the names; archive reads
#     a .strategy file, which is the fingerprint's own input. robot_profile reads config/robots/*.json only when an
#     example asks for a real robot's profile (`load_profile`), and profiles act on real robots
#     only, so no match result depends on them.
#   subprocess: robosim_wrapper starts robosim_subprocess.py in the robosim pixi env ->
#     SUBPROCESS_SCRIPTS, and environment_pin's robosim env listing and binary hashes.
#     game_log_writer asks git for the commit it records, only for a camera run's game log
#     (never in rsim), so no match result depends on it.
#   dynamic import: catalog.builder imports the strategy module of a name -> that module is the
#     fingerprint's own input.
#   module-level state: code in the closure, so already hashed; its values must not carry
#     from one match to the next in a worker, which the cache's replay spot-check verifies.
#     possession/shield state: reset in StrategyRunner.__init__; the planner's _PERP_ROTATIONS: a memo of
#     pure values; robosim_wrapper's idle sims: a fresh native world per start.
AUDIT = {
    "environment": [
        "utama_core.bundle.archive",
        "utama_core.bundle.strategy_file",
        "utama_core.config.settings",
        "utama_core.simulator.simulators.robosim.robosim_wrapper",
    ],
    "dynamic import": ["utama_core.strategy.catalog"],
    "module-level state": [
        "utama_core.motion_planning.fastpathplanning.planner",
        "utama_core.simulator.simulators.robosim.robosim_wrapper",
        "utama_core.geometry.pass_and_score_geometry",
        "utama_core.skills.shielding",
        "utama_core.strategy.pickers",
    ],
    "opens files": [
        "utama_core.bundle.archive",
        "utama_core.bundle.strategy_file",
        "utama_core.referee.profiles.profile_loader",
        "utama_core.engine.match_log",
        "utama_core.engine.match_stats",
        "utama_core.replay.columnar_reader",
        "utama_core.replay.columnar_writer",
        "utama_core.replay.game_log_writer",
        "utama_core.replay.replay_writer",
        "utama_core.strategy.catalog",
        "utama_core.team_controller.controllers.common.robot_profile",
    ],
    "subprocess": [
        "utama_core.bundle.strategy_file",
        "utama_core.replay.game_log_writer",
        "utama_core.simulator.simulators.robosim.robosim_wrapper",
    ],
}


def test_every_reach_outside_the_import_graph_is_accounted_for(graph):
    everything = graph.base_modules([MATCH_ENTRY]) | {m for c in _players() for m in graph.strategy_modules(c)}
    assert graph.audit(everything) == AUDIT


_TACTIC = "utama_core/tactics/defend.py"
_TACTIC_NAME = "utama_core.tactics.defend"
_TACTIC_SRC = '"""Old module doc."""\n\nX = 1\n\n\ndef f(a, b):\n    """Old doc."""\n    return a + b\n'


def _file_digest(root: Path, text: str) -> str:
    (root / _TACTIC).write_text(text)
    (row,) = CodeGraph(root).file_digests([_TACTIC_NAME])
    return row


def test_comments_docstrings_and_formatting_do_not_change_a_file_fingerprint(repo):
    before = _file_digest(repo, _TACTIC_SRC)
    edited = (
        '"""New module doc, longer."""\n# a new comment\nX = 1  # trailing comment\n\n\n'
        'def f(a,\n      b):\n    """New doc\n    spanning lines."""\n    return (a + b)\n'
    )
    assert _file_digest(repo, edited) == before


def test_changing_a_constant_changes_the_file_fingerprint(repo):
    before = _file_digest(repo, _TACTIC_SRC)
    assert _file_digest(repo, _TACTIC_SRC.replace("X = 1", "X = 2")) != before
