"""The rule's error rate, on simulated results (`tools/evaluation/verdict_sim.py`): the docs claim an
equal strategy is called "better" at most ALPHA of the time. Changing the rule or `ratings` is judged
against this."""

from __future__ import annotations

import pytest

from tools.evaluation import verdict_sim
from utama_core.analysis import ratings


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("n_opponents", [4, 5])
def test_an_equal_strategy_is_called_better_at_most_alpha_of_the_time(rho, n_opponents):
    r = verdict_sim.simulate(0.0, rho, n_opponents, trials=1500, seed=1)
    # the screen drops about half of them first, so the rate is well under ALPHA; the slack is for 1500 trials
    assert r["better"] <= ratings.ALPHA + 0.01


def test_a_clearly_better_small_edit_is_found():
    # one goal per match better, results correlated like a small edit of one strategy
    r = verdict_sim.simulate(1.0, 0.9, 5, trials=500, seed=2)
    assert r["better"] >= 0.8


def test_a_worse_strategy_is_dropped_in_the_screen():
    r = verdict_sim.simulate(-1.0, 0.9, 5, trials=300, seed=3)
    assert r["better"] == 0.0
    assert r["rounds"] < 3


def test_the_screen_never_accepts():
    perfect = ratings.paired_test([1.0, 2.0, 1.0, 3.0] * 10)
    assert ratings.decide(perfect, ratings.ROUNDS, confirming=False) == "confirm"
    assert ratings.decide(perfect, ratings.ROUNDS, confirming=True) == "better"


def test_a_stage_drops_only_from_its_second_round_and_at_a_mean_of_zero():
    zero = ratings.paired_test([1.0, -1.0, 0.0, 0.0])
    assert ratings.decide(zero, 1, confirming=False) is None
    assert ratings.decide(zero, 2, confirming=False) == "dropped"
    assert ratings.decide(zero, 2, confirming=True) == "dropped"


def test_the_confirm_stage_is_undecided_above_alpha():
    weak = ratings.paired_test([1.0, -1.0, 0.0, 0.5])
    assert weak.mean > 0 and weak.p > ratings.ALPHA
    assert ratings.decide(weak, ratings.ROUNDS, confirming=True) == "undecided"
