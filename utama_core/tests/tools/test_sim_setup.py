"""`tools/sim_setup.py`: which names load which strategy, and the runner it sets up."""

import pytest

from tools import sim_setup
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.referee.referee_command import RefereeCommand


@pytest.mark.parametrize(
    "name, expected",
    [
        ("scratch/try.py:build_x", True),
        ("/abs/path/try.py:build", True),
        ("split_shape", False),
        ("try.py:", False),  # no factory named
        ("try.txt:build", False),
        ("a.strategy", False),
    ],
)
def test_is_file_spec_needs_a_py_path_and_a_factory(name, expected):
    assert sim_setup.is_file_spec(name) is expected


def test_a_catalog_name_loads_an_abstract_strategy():
    assert isinstance(sim_setup.load_side("split_shape"), AbstractStrategy)


def test_an_unknown_name_lists_the_catalog():
    with pytest.raises(sim_setup.UnknownStrategy, match="unknown strategy 'nope'; strategies: .*split_shape"):
        sim_setup.load_side("nope")


def test_a_missing_strategy_file_is_named_not_treated_as_a_catalog_name(tmp_path):
    with pytest.raises(sim_setup.UnknownStrategy, match="no such strategy file"):
        sim_setup.load_side(str(tmp_path / "gone.strategy"))


def test_a_scratch_factory_gets_the_outfield_ids_first(tmp_path):
    scratch = tmp_path / "try.py"
    scratch.write_text("def build_x(ids, mc):\n    return ids, mc\n")
    assert sim_setup.factory_for(f"{scratch}:build_x")((1, 2))("mc") == ((1, 2), "mc")


def test_a_scratch_file_without_the_factory_is_named(tmp_path):
    scratch = tmp_path / "try.py"
    scratch.write_text("x = 1\n")
    with pytest.raises(sim_setup.UnknownStrategy, match="has no factory 'build_x'"):
        sim_setup.factory_for(f"{scratch}:build_x")


def test_sim_runner_plays_us_yellow_on_the_right_with_six_a_side():
    runner, referee = sim_setup.sim_runner(
        sim_setup.load_side("split_shape"),
        sim_setup.load_side("high_press"),
        initial_command=RefereeCommand.STOP,
    )
    try:
        assert runner.my_team_is_yellow and runner.my_team_is_right
        assert runner.exp_friendly == runner.exp_enemy == sim_setup.N_ROBOTS == 6
        assert runner.referee is referee
    finally:
        runner.close()
