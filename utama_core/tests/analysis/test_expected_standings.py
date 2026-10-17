import pytest

from utama_core.analysis.expected_standings import expected_standings, strengths


def _result(a: str, b: str, score_a: int, score_b: int, shots_a: int, shots_b: int) -> dict:
    return {
        "config_a": a,
        "config_b": b,
        "score_a": score_a,
        "score_b": score_b,
        "stats": {"shots": {"friendly": shots_a, "enemy": shots_b}},
    }


def test_strengths_recover_an_additive_model_and_sum_to_zero():
    """Margins that are exactly strength(a) - strength(b) come back exactly, centred on zero."""
    true = {"x": 2.0, "y": 0.0, "z": -2.0}
    results = [
        _result(
            a,
            b,
            0,
            0,
            int(true[a] - true[b]) if true[a] > true[b] else 0,
            int(true[b] - true[a]) if true[b] > true[a] else 0,
        )
        for a, b in (("x", "y"), ("z", "x"), ("y", "z"))
    ]
    s = strengths(results, "shots")
    assert s == pytest.approx(true)
    assert sum(s.values()) == pytest.approx(0.0)


def test_config_b_is_scored_from_the_enemy_side():
    """Stats are config_a's view: a match config_b dominates gives config_b the positive strength."""
    table = expected_standings([_result("x", "y", 0, 3, 1, 7)])
    assert table["y"] == {"goals": 1.5, "shots": 3.0}
    assert table["x"] == {"goals": -1.5, "shots": -3.0}


def test_estimators_can_disagree():
    """A strategy that outshoots its opponent but loses on goals ranks high on shots, low on goals."""
    table = expected_standings([_result("x", "y", 0, 1, 6, 1)])
    assert table["x"]["shots"] > 0 > table["x"]["goals"]


def test_no_results_gives_empty_table():
    assert expected_standings([]) == {}
