import pytest

from tools.evaluation import evaluate
from utama_core.analysis import ratings


def test_every_round_is_every_pair_once_under_its_seed():
    for k in (1, 2):
        fixtures = evaluate.round_fixtures(k, ["c"], "ref", ["o1", "o2"])
        assert len(fixtures) == 6 and {f[2] for f in fixtures} == {k}
        assert {frozenset(f[:2]) for f in fixtures} == {
            frozenset(p) for p in [("c", "ref"), ("c", "o1"), ("c", "o2"), ("ref", "o1"), ("ref", "o2"), ("o1", "o2")]
        }


def test_candidate_and_reference_meet_each_opponent_with_the_same_colour():
    for k in (1, 2, 3):
        fixtures = evaluate.round_fixtures(k, ["c"], "ref", ["o"])
        c_is_a = ("c", "o", k) in fixtures
        assert c_is_a == (k % 2 == 1)
        assert (("ref", "o", k) in fixtures) == c_is_a


def test_the_confirm_stage_plays_fresh_seeds_and_decides_on_them_alone(monkeypatch):
    monkeypatch.setattr(ratings, "ROUNDS", 3)
    assert [evaluate.seed_of(k) for k in range(1, 7)] == [1, 2, 3, 101, 102, 103]
    assert evaluate.stage_seeds(2) == {1, 2}
    assert evaluate.stage_seeds(5) == {101, 102}


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


def test_a_decision_uses_its_stages_seeds_only():
    ledger = {"1": {"seed": 1}, "2": {"seed": 2}, "101": {"seed": 101}, "x": {"seed": None}}
    assert list(evaluate._with_seeds(ledger, {1, 2})) == ["1", "2"]
    assert list(evaluate._with_seeds(ledger, {101})) == ["101"]


def test_colours_follow_the_seed_not_the_round():
    # a confirm seed is played in round ROUNDS+1 or later, so a run with another ROUNDS plays it in a
    # round of the other parity: the store would then hold it in both colours, and the verdict would
    # count both (test_the_verdict_does_not_depend_on_other_seeds_already_in_the_store)
    for k in (1, 2, 7, 8):
        assert ("c", "o", 101) in evaluate.round_fixtures(k, ["c"], "ref", ["o"], seed=101)
        assert ("o", "c", 102) in evaluate.round_fixtures(k, ["c"], "ref", ["o"], seed=102)


def test_the_players_load_and_the_core_is_in_the_catalog():
    from tools.evaluation.match import _CONFIG_NAMES
    from tools.evaluation.sim_settings import load_players

    players = load_players()
    assert players["champion"] in players["core"]
    assert set(players["core"]) <= set(_CONFIG_NAMES)


def test_the_reference_world_and_robots_make_one_valid_sim_noise():
    from tools.evaluation.sim_settings import reference_noise
    from utama_core.simulator.utils.sim_noise import SimNoise

    noise = SimNoise.from_dict(reference_noise(seed=3))
    assert noise.seed == 3
    assert noise.latency_estimate_s == pytest.approx(0.07)


def test_every_reference_setting_names_its_source():
    import json

    from tools.evaluation.sim_settings import REFERENCE_DIR

    for name in ("world", "robots"):
        data = json.loads((REFERENCE_DIR / f"{name}.json").read_text())
        assert set(data["sources"]) == set(data["values"]), name


def test_the_reference_sim_does_not_draw_robot_motor_limits():
    """Acceleration is the command cap; a per-match draw would be a robot capability as noise."""
    from tools.evaluation.sim_settings import reference_noise
    from utama_core.simulator.utils.sim_noise import SimNoise

    assert not SimNoise.from_dict(reference_noise()).motion_limits


FPS = {"c": "fc", "o": "fo", "ch": "fch"}


def _ledger_row(a, b, stalls, fixture, seed=1, a_fp=None):
    return {
        "fixture": fixture,
        "a": a,
        "b": b,
        "a_fp": a_fp or FPS[a],
        "b_fp": FPS[b],
        "seed": seed,
        "stalls": stalls,
    }


def test_stall_coverage_counts_only_matches_where_the_candidate_was_config_a():
    ledger = {
        "1": _ledger_row("c", "o", [], "1"),  # checked, no stalls
        "2": _ledger_row("o", "c", None, "2"),  # candidate as config_b: unchecked
        "3": _ledger_row("c", "o", None, "3", seed=2),  # reused old line: unknown, unchecked
        "4": _ledger_row("o", "c", [{"kind": "RESTART_STALL", "sim_time": 5.0}], "4", seed=2),
    }
    cov = evaluate.stall_coverage(ledger, FPS, "c")
    assert (cov["checked"], cov["total"]) == (1, 4)
    assert cov["stalls"] == []


def test_stall_coverage_lists_each_stall_with_its_match_seed_kind_and_time():
    ledger = {
        "1": _ledger_row("c", "o", [{"kind": "RESTART_STALL", "sim_time": 5.0, "diagnosis": "ball in goal"}], "1")
    }
    cov = evaluate.stall_coverage(ledger, FPS, "c")
    assert cov["checked"] == 1 and cov["total"] == 1
    assert cov["stalls"] == [
        {"a": "c", "b": "o", "seed": 1, "kind": "RESTART_STALL", "sim_time": 5.0, "diagnosis": "ball in goal"}
    ]


def test_stall_coverage_ignores_matches_at_other_code_and_other_players():
    ledger = {
        "1": _ledger_row("c", "o", [], "1", a_fp="old"),  # the candidate's code has changed since
        "2": _ledger_row("ch", "o", [], "2", a_fp="fch"),  # the champion's match, not the candidate's
    }
    assert evaluate.stall_coverage(ledger, FPS, "c") == {"checked": 0, "total": 0, "stalls": []}


def test_stall_lines_say_none_only_when_checked_and_unknown_when_not():
    assert evaluate.stall_lines({"checked": 3, "total": 7, "stalls": []}, "candidate") == [
        "stalls checked in 3 of 7 candidate matches (watchdog covers config_a only): none"
    ]
    assert evaluate.stall_lines({"checked": 0, "total": 7, "stalls": []}, "candidate")[0].endswith(": unknown")
