import pytest
from scipy import stats

from utama_core.analysis import ratings


def test_identical_code_gives_no_evidence():
    # identical code plays identical matches: every diff is exactly 0
    assert ratings.paired_test([0.0] * 30).p == 1.0


def test_too_few_diffs_give_no_evidence():
    assert ratings.paired_test([]).p == 1.0
    assert ratings.paired_test([3.0]).p == 1.0


def test_the_p_value_is_the_one_sided_t_test():
    diffs = [1.0, 0.0, 1.0, 1.0, 1.0]
    t = ratings.paired_test(diffs)
    assert t.mean == pytest.approx(0.8)
    assert t.se == pytest.approx(0.2)  # sample SD 0.447 over sqrt(5)
    assert t.p == pytest.approx(stats.ttest_1samp(diffs, 0.0, alternative="greater").pvalue)


def test_a_worse_candidate_has_a_large_p():
    assert ratings.paired_test([-1.0, -2.0, 0.0, -1.0]).p > 0.9


def test_a_repeated_exact_margin_is_decisive_either_way():
    assert ratings.paired_test([1.0, 1.0, 1.0]).p == 0.0
    assert ratings.paired_test([-1.0, -1.0, -1.0]).p == 1.0
