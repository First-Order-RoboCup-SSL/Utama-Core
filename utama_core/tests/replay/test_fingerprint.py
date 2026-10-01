"""`utama_core.replay.fingerprint`: which edits change which fingerprints.

Most tests build a small repo in `tmp_path` with the real layout (a `kernel_strategy.py`
with factories, tactic modules, a planner the runner imports, `tournament_lib.py`), edit
one file, and check exactly the fingerprints that should move do. The rest run against
this repo itself."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from utama_core.replay import fingerprint as fp
from utama_core.replay.fingerprint import (
    BENCH_ENTRIES,
    KERNEL_STRATEGY,
    REPO_ROOT,
    ROUND_ROBIN_ENTRY,
    CodeGraph,
    config_names,
    match_key,
)

KERNEL = '''\
"""Factories."""

from __future__ import annotations

from utama_core.tactics.attack import AttackTactic
from utama_core.tactics.defend import DefendTactic
from utama_core.tactics.unused import UnusedTactic

_MARGIN = 0.05  # used by the shared helper
_ONLY_B = 3
_streak = 0


def _shared_helper(x):
    return x + _MARGIN


def _b_picker(x):
    global _streak
    _streak += 1
    return _shared_helper(x) * _ONLY_B


def build_a_kernel_strategy(ids):
    return [AttackTactic(), _shared_helper(1)]


def build_b_kernel_strategy(ids):
    return [DefendTactic(), _b_picker(2)]


def build_c_kernel_strategy(ids):
    # no helper, no tactic of its own
    return [len(ids)]
'''

FILES = {
    "tournament_lib.py": "from utama_core.run.runner import run\nfrom utama_core.strategy import kernel_strategy\n",
    "utama_core/__init__.py": "",
    "utama_core/run/__init__.py": "",
    "utama_core/run/runner.py": "from ..planning import planner\nfrom utama_core.profiles import loader\n\n\ndef run():\n"
    "    return planner.plan()\n",
    "utama_core/planning/__init__.py": "",
    "utama_core/planning/planner.py": "def plan():\n    return 1\n",
    "utama_core/profiles/__init__.py": "",
    "utama_core/profiles/loader.py": "def load():\n    return open('sim.yaml').read()\n",
    "utama_core/profiles/sim.yaml": "speed: 1\n",
    "utama_core/strategy/__init__.py": "",
    "utama_core/strategy/kernel_strategy.py": KERNEL,
    "utama_core/tactics/__init__.py": "",
    "utama_core/tactics/attack.py": "from utama_core.skills.kick import kick\n\n\nclass AttackTactic:\n"
    "    def tick(self):\n        return kick()\n",
    # `kick` imports its helper inside the function: still in the closure
    "utama_core/skills/kick.py": "def kick():\n    from utama_core.skills.aim import aim\n\n    return aim()\n",
    "utama_core/skills/aim.py": "def aim():\n    return 0\n",
    "utama_core/tactics/defend.py": "class DefendTactic:\n    pass\n",
    "utama_core/tactics/unused.py": 'class UnusedTactic:\n    pass\n\n\nif __name__ == "__main__":\n    print(1)\n',
    "utama_core/rsoccer_simulator/src/Simulators/robosim/robosim_subprocess.py": "import json\n",
}
CONFIGS = ["build_a_kernel_strategy", "build_b_kernel_strategy", "build_c_kernel_strategy"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for pkg in ("utama_core/rsoccer_simulator", "utama_core/rsoccer_simulator/src"):
        (tmp_path / pkg / "__init__.py").write_text("")
    return tmp_path


def _fingerprints(root: Path) -> dict[str, str]:
    g = CodeGraph(root)
    out = {c: g.strategy_fingerprint(c) for c in CONFIGS}
    out["base"] = g.base_fingerprint([ROUND_ROBIN_ENTRY])
    return out


def _changed_after(root: Path, rel: str, old: str, new: str) -> set[str]:
    before = _fingerprints(root)
    path = root / rel
    text = path.read_text()
    assert old in text, f"{old!r} not in {rel}"
    path.write_text(text.replace(old, new, 1))
    after = _fingerprints(root)
    return {k for k in before if before[k] != after[k]}


def _short(changed: set[str]) -> set[str]:
    return {c.removeprefix("build_").removesuffix("_kernel_strategy") for c in changed}


KS = "utama_core/strategy/kernel_strategy.py"


def test_editing_one_factory_changes_only_that_config(repo):
    assert _short(_changed_after(repo, KS, "return [len(ids)]", "return [len(ids) + 1]")) == {"c"}


def test_editing_a_helper_changes_the_configs_that_reach_it(repo):
    assert _short(_changed_after(repo, KS, "return x + _MARGIN", "return x - _MARGIN")) == {"a", "b"}


def test_editing_a_constant_changes_the_configs_that_reach_it_through_helpers(repo):
    assert _short(_changed_after(repo, KS, "_ONLY_B = 3", "_ONLY_B = 4")) == {"b"}
    assert _short(_changed_after(repo, KS, "_MARGIN = 0.05", "_MARGIN = 0.06")) == {"a", "b"}


def test_module_state_named_by_global_is_in_the_slice(repo):
    assert _short(_changed_after(repo, KS, "_streak = 0", "_streak = 5")) == {"b"}


def test_a_comment_or_docstring_outside_every_slice_changes_nothing(repo):
    assert _changed_after(repo, KS, '"""Factories."""', '"""Factories, edited."""') == set()
    assert _changed_after(repo, KS, "# used by the shared helper", "# edited") == set()


def test_a_future_import_is_in_every_slice(repo):
    changed = _changed_after(repo, KS, "from __future__ import annotations\n", "")
    assert _short(changed) == {"a", "b", "c"}


def test_editing_a_tactic_changes_exactly_the_configs_that_use_it(repo):
    rel = "utama_core/tactics/attack.py"
    assert _short(_changed_after(repo, rel, "return kick()", "return kick() + 1")) == {"a"}


def test_a_function_level_import_is_followed(repo):
    assert _short(_changed_after(repo, "utama_core/skills/aim.py", "return 0", "return 1")) == {"a"}


def test_a_tactic_no_config_uses_changes_nothing(repo):
    assert _changed_after(repo, "utama_core/tactics/unused.py", "pass", "x = 1") == set()


def test_import_time_effects_move_a_module_into_the_base(repo):
    # `unused` runs in every match (kernel_strategy imports it): once its import-time code
    # can reach outside it, it is part of what every match runs.
    rel = "utama_core/tactics/unused.py"
    assert _changed_after(
        repo, rel, "class UnusedTactic:", "import sys\nsys.path.append('x')\n\n\nclass UnusedTactic:"
    ) == {"base"}
    assert "utama_core.tactics.unused" in CodeGraph(repo).ambient_modules()
    assert _changed_after(repo, rel, "pass", "y = 2") == {"base"}


def test_a_main_guard_is_not_an_import_time_effect(repo):
    assert CodeGraph(repo).ambient_modules() == {}


def test_editing_shared_infrastructure_changes_the_base_and_every_match_key(repo):
    g = CodeGraph(repo)
    keys = {(a, b): match_key(g, a, b, duration_seconds=65.0) for a in CONFIGS for b in CONFIGS if a < b}
    changed = _changed_after(repo, "utama_core/planning/planner.py", "return 1", "return 2")
    assert changed == {"base"}
    g = CodeGraph(repo)
    assert all(match_key(g, a, b, duration_seconds=65.0) != k for (a, b), k in keys.items())


def test_a_star_import_is_in_every_slice(repo):
    path = repo / KS
    path.write_text(path.read_text().replace("import UnusedTactic", "import *"))
    changed = _changed_after(repo, "utama_core/tactics/unused.py", "pass", "x = 1")
    assert _short(changed) == {"a", "b", "c"}


def test_shared_code_importing_kernel_strategy_puts_the_whole_file_in_the_base(repo):
    rel = "utama_core/planning/planner.py"
    path = repo / rel
    path.write_text("from utama_core.strategy.kernel_strategy import _shared_helper\n\n\n" + path.read_text())
    assert _changed_after(repo, KS, "_ONLY_B = 3", "_ONLY_B = 4") == {"base", "build_b_kernel_strategy"}
    assert _changed_after(repo, KS, "# used by the shared helper", "# edited") == {"base"}


def test_the_rsim_subprocess_script_and_data_files_are_in_the_base(repo):
    rel = "utama_core/rsoccer_simulator/src/Simulators/robosim/robosim_subprocess.py"
    assert _changed_after(repo, rel, "import json", "import json, struct") == {"base"}
    assert _changed_after(repo, "utama_core/profiles/sim.yaml", "speed: 1", "speed: 2") == {"base"}


def test_fingerprints_do_not_depend_on_where_the_checkout_is(repo, tmp_path_factory):
    import shutil

    other = tmp_path_factory.mktemp("worktree") / "elsewhere"
    shutil.copytree(repo, other)
    (other / ".git").write_text("gitdir: /some/other/checkout\n")
    assert _fingerprints(other) == _fingerprints(repo)


def test_a_relative_import_resolves(repo):
    g = CodeGraph(repo)
    assert "utama_core.planning.planner" in g.closure(["utama_core.run.runner"])


def test_match_key_covers_every_setting(repo):
    g = CodeGraph(repo)
    a, b = CONFIGS[:2]
    ref = match_key(g, a, b, duration_seconds=65.0)
    variants = [
        match_key(g, b, a, duration_seconds=65.0),
        match_key(g, a, b, duration_seconds=20.0),
        match_key(g, a, b, duration_seconds=65.0, a_is_right=False),
        match_key(g, a, b, duration_seconds=65.0, a_kicks_off=False),
        match_key(g, a, b, duration_seconds=65.0, control_scheme="dwa"),
        match_key(g, a, b, duration_seconds=65.0, fuzz_seed=1),
        match_key(g, a, b, duration_seconds=65.0, recorded=False),
    ]
    assert len({ref, *variants}) == 1 + len(variants)
    # the fuzz interval only matters when fuzzing is on
    assert match_key(g, a, b, duration_seconds=65.0, fuzz_interval_s=(1.0, 2.0)) == ref
    assert match_key(g, a, b, duration_seconds=65.0, fuzz_seed=1, fuzz_interval_s=(1.0, 2.0)) != variants[5]


def test_numerics_env_vars_change_the_base(repo, monkeypatch):
    monkeypatch.delenv("UTAMA_EXACT_MATH", raising=False)
    before = CodeGraph(repo).base_fingerprint([ROUND_ROBIN_ENTRY])
    monkeypatch.setenv("UTAMA_EXACT_MATH", "1")
    assert CodeGraph(repo).base_fingerprint([ROUND_ROBIN_ENTRY]) != before
    with_exact = CodeGraph(repo).base_fingerprint([ROUND_ROBIN_ENTRY])
    monkeypatch.setenv("UNRELATED_VAR", "1")
    assert CodeGraph(repo).base_fingerprint([ROUND_ROBIN_ENTRY]) == with_exact


# --- this repo ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph() -> CodeGraph:
    return CodeGraph(REPO_ROOT)


def test_configs_found_from_source_match_the_round_robin(graph):
    import tournament_lib

    assert config_names(graph) == sorted(tournament_lib._CONFIG_NAMES)


def test_every_tactic_a_factory_builds_is_in_its_closure(graph):
    """Independent of the static analysis: build every config's `Strategy` and check the
    module of each tactic object it holds is in that config's fingerprinted modules."""
    from utama_core.strategy import kernel_strategy

    for config in config_names(graph):
        strategy = getattr(kernel_strategy, config)((1, 2, 3, 4, 5))(None)
        modules = {type(t).__module__ for t in strategy._tactics.values()}
        allowed = graph.strategy_modules(config) | graph.base_modules([ROUND_ROBIN_ENTRY])
        assert modules <= allowed, (config, modules - allowed)


def test_the_tactics_kernel_strategy_imports_are_outside_the_base(graph):
    """Otherwise a tactic edit would rerun every match: the per-config split is the point."""
    base = graph.base_modules([ROUND_ROBIN_ENTRY]) | graph.base_modules(BENCH_ENTRIES)
    for name in ("give_and_go", "press_and_contain", "shadow_and_mark", "switch_of_play", "defense"):
        assert f"utama_core.tactics.{name}" not in base


def test_the_base_holds_the_sim_script_and_referee_profiles(graph):
    base = graph.base_modules([ROUND_ROBIN_ENTRY])
    assert set(fp.SUBPROCESS_SCRIPTS) <= base
    assert any("custom_referee/profiles/simulation.yaml" in row for row in graph.data_files(base))


# Every place the code a match runs reaches outside the import graph, and how each is
# handled. A new entry here fails this test: decide how it enters the fingerprint (or why
# it doesn't need to), then add it.
#   environment: settings.py reads UTAMA_EXACT_MATH -> ENV_PREFIXES; robosim_wrapper copies
#     the environment into the sim subprocess -> ENV_PREFIXES covers the numerics ones.
#   opens files: profile_loader reads profiles/*.yaml -> data_files. bench_scenario reads a
#     bank, whose start is itself in the bench key. The rest write a match's outputs (match
#     log, stats, replays) or read old ones for harvesting, never during play.
#   subprocess: robosim_wrapper starts robosim_subprocess.py in the robosim pixi env ->
#     SUBPROCESS_SCRIPTS, and environment_pin's robosim env listing and binary hashes.
#   getattr on a module by computed name: tournament_lib and scenario_scorer look a factory
#     up by name -> the config names are the fingerprint's own inputs.
#   module-level state: code in the closure, so already hashed; its values must not carry
#     from one match to the next in a worker, which the cache's replay spot-check verifies.
#     possession/shield state: reset in StrategyRunner.__init__; kernel_strategy's
#     _possession_streak: reset by its factory; the planner's _PERP_ROTATIONS: a memo of
#     pure values; robosim_wrapper's idle sims: a fresh native world per start.
AUDIT = {
    "environment": [
        "utama_core.config.settings",
        "utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper",
    ],
    "getattr on a module by computed name": ["tournament_lib", "utama_core.replay.scenario_scorer"],
    "module-level state": [
        "utama_core.motion_planning.src.fastpathplanning.planner",
        "utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper",
        "utama_core.shared.pass_and_score_geometry",
        "utama_core.skills.src.shielding",
        "utama_core.strategy.kernel_strategy",
    ],
    "opens files": [
        "utama_core.custom_referee.profiles.profile_loader",
        "utama_core.engine.match_log",
        "utama_core.engine.match_stats",
        "utama_core.replay.bench_scenario",
        "utama_core.replay.columnar_reader",
        "utama_core.replay.columnar_writer",
        "utama_core.replay.replay_player",
        "utama_core.replay.replay_writer",
        "utama_core.replay.scenario",
        "utama_core.replay.turnover_breakdown",
    ],
    "subprocess": ["utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper"],
}


def test_every_reach_outside_the_import_graph_is_accounted_for(graph):
    everything = (
        graph.base_modules([ROUND_ROBIN_ENTRY]) | graph.base_modules(BENCH_ENTRIES) | graph.closure([KERNEL_STRATEGY])
    )
    assert graph.audit(everything) == AUDIT
