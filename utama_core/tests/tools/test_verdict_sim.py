"""The stopping rule's error rate, on simulated results (`tools/evaluation/verdict_sim.py`): the docs
claim two equal strategies are called "better" rarely. Changing the rule or `ratings` is judged
against this."""

from __future__ import annotations

import pytest

from tools.evaluation import verdict_sim


def test_the_simulation_plays_evaluate_pys_rule():
    from tools.evaluation import evaluate

    assert (verdict_sim.PROMOTE, verdict_sim.DROP, verdict_sim.MAX_ROUNDS) == (
        evaluate.PROMOTE,
        evaluate.DROP,
        evaluate.MAX_ROUNDS,
    )


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("n_opponents", [4, 5])
def test_an_equal_strategy_is_rarely_called_better(rho, n_opponents):
    r = verdict_sim.simulate(0.0, rho, n_opponents, trials=1500, seed=1)
    assert r["better"] <= 0.05


def test_a_clearly_better_small_edit_is_found():
    # one goal per match better, results correlated like a small edit of one strategy
    r = verdict_sim.simulate(1.0, 0.9, 5, trials=500, seed=2)
    assert r["better"] >= 0.8
