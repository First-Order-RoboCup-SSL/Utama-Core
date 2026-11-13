"""The benchmark runs any registered planner by name (tutorials/first_planner.md), not only its
three defaults; each keeps one colour in every figure."""

import sys

from tools import motion_planning_benchmark as benchmark


def test_every_registered_scheme_can_be_named_and_the_defaults_stay_the_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--schemes", "pid", "fpp"])
    assert benchmark.parse_args().schemes == ["pid", "fpp"]
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    assert benchmark.parse_args().schemes == ["fpp", "dwa", "trajsample"]


def test_each_scheme_has_its_own_colour_and_the_defaults_keep_theirs():
    colours = [benchmark._colour(s) for s in benchmark.ALL_SCHEMES]
    assert len(set(colours)) == len(colours)
    assert [benchmark._colour(s) for s in benchmark.SCHEMES] == list(benchmark.SLOTS[:3])
