from tools.evaluation import evaluate
from utama_core.replay import eval_ledger


def test_round_fixtures_pair_candidate_and_champion_on_the_same_opponents_and_seed():
    fixtures = evaluate.round_fixtures(["c"], "ch", ["o1", "o2"], 1)
    assert set(fixtures) == {("c", "o1", 1), ("c", "o2", 1), ("c", "ch", 1), ("ch", "o1", 1), ("ch", "o2", 1)}


def test_round_fixtures_swap_sides_in_even_rounds():
    assert ("o1", "c", 2) in evaluate.round_fixtures(["c"], "ch", ["o1"], 2)
    assert ("ch", "c", 2) in evaluate.round_fixtures(["c"], "ch", ["o1"], 2)


def test_two_candidates_share_the_champions_fixtures():
    fixtures = evaluate.round_fixtures(["c1", "c2"], "ch", ["o1"], 1)
    assert fixtures.count(("ch", "o1", 1)) == 1
    assert ("c1", "c2", 1) not in fixtures


def test_verdict_thresholds_are_inclusive_and_cap_rounds():
    assert evaluate.verdict(evaluate.PROMOTE, 1, 6) == "better"
    assert evaluate.verdict(evaluate.DROP, 1, 6) == "worse"
    assert evaluate.verdict(0.5, 5, 6) is None
    assert evaluate.verdict(0.5, 6, 6) == "undecided"


def test_ledger_round_trip_keeps_the_last_entry_per_fixture_and_skips_torn_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    base = {"env": "e", "a": "x", "b": "y", "a_fp": "1", "b_fp": "2", "seed": 1}
    eval_ledger.append([{**base, "fixture": "f", "score_a": 1}, {**base, "env": "other", "fixture": "g"}], path)
    eval_ledger.append([{**base, "fixture": "f", "score_a": 2}], path)
    with open(path, "a") as f:
        f.write('{"env": "e", "fixt')
    loaded = eval_ledger.load("e", path)
    assert list(loaded) == ["f"] and loaded["f"]["score_a"] == 2


def test_fixture_id_depends_on_seed_and_code():
    f = eval_ledger.fixture_id
    assert f("e", "x", "1", "y", "2", 1) != f("e", "x", "1", "y", "2", 2)
    assert f("e", "x", "1", "y", "2", 1) != f("e", "x", "9", "y", "2", 1)


def test_per_opponent_compares_only_matches_with_the_same_seed_and_colour():
    """The champion's seed-1 match as config_a pairs with the candidate's seed-1 match as
    config_a; the candidate's seed-1 match as config_b has no partner and is left out."""

    def e(a, b, sa, sb, seed):
        return {"a": a, "b": b, "a_fp": "f", "b_fp": "f", "score_a": sa, "score_b": sb, "seed": seed}

    ledger = {
        "1": e("c", "o", 2, 0, 1),
        "2": e("o", "c", 5, 0, 1),
        "3": e("ch", "o", 1, 0, 1),
        "4": e("c", "ch", 0, 3, 1),
    }
    rows = evaluate._per_opponent(ledger, {"c": "f", "o": "f", "ch": "f"}, "c", "ch", ["o"])
    assert rows["o"] == {"candidate": 2, "champion": 1, "matches": 1}
    assert rows["head-to-head"]["candidate"] == -3


def test_paired_differences_use_same_seed_and_colour_and_orient_head_to_head():
    def e(a, b, sa, sb, seed):
        return {"a": a, "b": b, "a_fp": "f", "b_fp": "f", "score_a": sa, "score_b": sb, "seed": seed}

    ledger = {
        "1": e("c", "o", 2, 0, 1),  # pairs with 3
        "2": e("o", "c", 5, 0, 1),  # candidate as config_b: no champion partner
        "3": e("ch", "o", 1, 0, 1),
        "4": e("ch", "c", 3, 0, 2),  # head-to-head, candidate as config_b
    }
    diffs, h2h = evaluate.paired(ledger, {"c": "f", "o": "f", "ch": "f"}, "c", "ch", ["o"])
    assert diffs == [2 - 1]
    assert h2h == [-3]


def test_a_verdict_after_round_k_ignores_later_seeds_already_in_the_ledger():
    ledger = {"1": {"seed": 1}, "2": {"seed": 2}, "3": {"seed": None}}
    assert list(evaluate._up_to_round(ledger, 1)) == ["1"]
