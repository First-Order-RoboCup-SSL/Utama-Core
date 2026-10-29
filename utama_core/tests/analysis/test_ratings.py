import numpy as np
import pytest

from utama_core.analysis import ratings


def test_no_matches_keeps_the_prior():
    r = ratings.fit([], names=["x", "y"])
    assert r.strength("x") == pytest.approx((0.0, ratings.PRIOR_SD))
    assert r.p_better("x", "y") == pytest.approx(0.5)


def test_many_matches_recover_true_strengths():
    """With plenty of noisy matches the posterior mean lands on the true margins and its SD
    shrinks like 1/sqrt(matches)."""
    rng = np.random.default_rng(0)
    true = {"x": 1.0, "y": 0.0, "z": -1.0}
    pairs = [("x", "y"), ("y", "z"), ("z", "x")]
    matches = [(a, b, true[a] - true[b] + rng.normal(0, 1.6)) for _ in range(400) for a, b in pairs]
    r = ratings.fit(matches)
    for n, s in true.items():
        assert r.strength(n)[0] == pytest.approx(s, abs=0.1)
    assert r.noise_sd == pytest.approx(1.6, abs=0.1)
    assert r.p_better("x", "z") > 0.999


def test_margin_is_shrunk_toward_zero_by_the_prior():
    """One 3-0 win is not a 3-goal strength gap: the prior pulls it in."""
    r = ratings.fit([("x", "y", 3.0)], noise_sd=1.6)
    m, sd = r.diff("x", "y")
    assert 0 < m < 3.0
    expected = 2 * ratings.PRIOR_SD**2 / (2 * ratings.PRIOR_SD**2 + 1.6**2) * 3.0
    assert m == pytest.approx(expected)


def test_noise_sd_cannot_collapse_on_few_matches():
    """Two matches that fit perfectly still leave noise_sd near its prior."""
    r = ratings.fit([("x", "y", 1.0), ("x", "y", 1.0)])
    assert r.noise_sd > 0.8 * ratings.NOISE_SD


def test_common_opponents_compare_two_strategies_that_never_met():
    """x beats o by more than y does, so x is likely stronger than y without x v y matches."""
    matches = [("x", "o", 2.0)] * 20 + [("y", "o", 0.0)] * 20
    r = ratings.fit(matches)
    assert r.p_better("x", "y") > 0.99


def test_identical_paired_results_pin_the_difference_near_zero():
    """The case that forced the paired test: a copy of the champion matches its results exactly
    against every opponent, and a few lucky head-to-head wins must not make it 'better'."""
    m, sd = ratings.paired_difference([0.0] * 24, [1, 4, 1, 0], noise_sd=1.9)
    assert abs(m) < 0.3
    assert ratings.p_positive(m, sd) < 0.9


def test_paired_spread_is_shrunk_toward_no_cancelling():
    """Two equal diffs have zero spread, but the variance stays near 2 * noise_sd^2 * N0 / (N0 + 1)."""
    m, sd = ratings.paired_difference([1.0, 1.0], [], noise_sd=1.9)
    var_d = ratings.PAIRED_PRIOR_N * 2 * 1.9**2 / (ratings.PAIRED_PRIOR_N + 1)
    assert sd == pytest.approx((1 / (2 * ratings.PRIOR_SD**2) + 2 / var_d) ** -0.5)


def test_consistent_paired_gains_are_decisive():
    m, sd = ratings.paired_difference([1.0, 2.0, 1.0, 1.5] * 6, [], noise_sd=1.9)
    assert ratings.p_positive(m, sd) > 0.99


def test_no_evidence_is_a_coin_flip():
    m, sd = ratings.paired_difference([], [], noise_sd=1.9)
    assert ratings.p_positive(m, sd) == pytest.approx(0.5)
