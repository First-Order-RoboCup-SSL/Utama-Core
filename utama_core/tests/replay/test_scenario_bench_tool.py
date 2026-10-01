import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

from utama_core.replay.bench_scenario import load_bank, save_bank
from utama_core.replay.hand_authored_scenarios import all_hand_authored_scenarios

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "scenario_bench.py"
_spec = importlib.util.spec_from_file_location("scenario_bench", _TOOL)
scenario_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scenario_bench)


def _moved(bench_scenario, scenario_id: str, dx: float):
    return dataclasses.replace(
        bench_scenario,
        scenario_id=scenario_id,
        scenario=dataclasses.replace(bench_scenario.scenario, ball_x=bench_scenario.scenario.ball_x + dx),
    )


def test_growing_a_bank_adds_every_new_harvested_start_and_drops_duplicates(tmp_path, monkeypatch):
    existing = list(all_hand_authored_scenarios())
    old_bank, new_bank = tmp_path / "bank_old.json", tmp_path / "bank_new.json"
    save_bank(existing, old_bank, bank_id="old")
    harvested = [
        _moved(existing[0], "copy_of_existing", 0.01),
        _moved(existing[0], "new_a", 1.0),
        _moved(existing[1], "new_b", 1.0),
    ]
    monkeypatch.setattr(scenario_bench, "harvest_run_dir", lambda *a, **k: (harvested, {}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scenario_bench.py",
            "--harvest-from",
            str(tmp_path),
            "--merge-into",
            str(old_bank),
            "--save-bank",
            str(new_bank),
            "--list-scenarios",
        ],
    )

    assert scenario_bench.main() == 0

    # the hand-authored scenarios the tool always loads, and the near-copy, duplicate the old bank
    bank_id, saved = load_bank(new_bank)
    assert bank_id == "bank_new"
    assert [s.scenario_id for s in saved] == [s.scenario_id for s in existing] + ["new_a", "new_b"]


def test_overall_line_reports_t_and_a_no_difference_run():
    rows = [{"delta": d} for d in (-0.5, -0.1, -0.3, 0.1)]
    assert "t -1.55" in scenario_bench._overall(rows)
    assert "the same on all 3 scenarios" in scenario_bench._overall([{"delta": 0.0}] * 3)
    assert scenario_bench._overall([{"delta": None}]) is None


def test_a_harvest_drops_duplicate_starts_even_without_a_bank_to_merge_into(tmp_path, monkeypatch):
    base = list(all_hand_authored_scenarios())[0]
    harvested = [_moved(base, "first", 1.0), _moved(base, "same_again", 1.01)]
    monkeypatch.setattr(scenario_bench, "harvest_run_dir", lambda *a, **k: (harvested, {}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scenario_bench.py",
            "--harvest-from",
            str(tmp_path),
            "--families",
            base.provenance.family.value,
            "--list-scenarios",
        ],
    )

    scenarios, merged, _ = scenario_bench._load_bank(scenario_bench.parse_args())

    assert "same_again" not in [s.scenario_id for s in scenarios]
    assert "first" in [s.scenario_id for s in scenarios]


def test_a_scenario_that_errored_on_either_side_is_left_out_not_scored_as_neutral(monkeypatch):
    # a start that fails to set up comes back NEUTRAL with an error; counting it as a
    # real NEUTRAL outcome would put a fake zero (or a fake difference) into the A/B
    ok, broken = list(all_hand_authored_scenarios())[:2]

    def fake_runs(bench_scenario, config, opponent, horizon_s, repeats, *_reuse):
        failed = bench_scenario.scenario_id == broken.scenario_id and config == "base"
        return {"outcomes": [0] * repeats, "fouls": 0, "stalls": 0, "errors": ["setup failed"] if failed else []}

    monkeypatch.setattr(scenario_bench, "_runs", fake_runs)
    rows, _ = scenario_bench._score(
        [ok, broken], candidate="cand", opponent="opp", horizon_s=1.0, repeats=1, baseline="base"
    )

    assert [r["delta"] for r in rows] == [0.0, None]


def test_an_earlier_run_s_errored_scenarios_are_not_a_baseline(tmp_path):
    path = tmp_path / "earlier.json"
    path.write_text(
        json.dumps(
            {
                "opponent": "opp",
                "horizon_s": 20.0,
                "repeats": 1,
                "results": [
                    {"scenario_id": "fine", "candidate_outcomes": [1], "candidate_errors": []},
                    {"scenario_id": "broken", "candidate_outcomes": [0], "candidate_errors": ["setup failed"]},
                ],
            }
        )
    )
    outcomes, _ = scenario_bench._load_against(path, opponent="opp", horizon_s=20.0, repeats=1)
    assert outcomes == {"fine": [1]}


def _many(n):
    base = list(all_hand_authored_scenarios())[0]
    return [_moved(base, f"s{i:03d}", 0.2 * i) for i in range(n)]


def _fake_runs_candidate_worse(bench_scenario, config, opponent, horizon_s, repeats, *_reuse):
    # the candidate loses the ball on 3 starts in 4, the baseline never does
    worse = config == "cand" and int(bench_scenario.scenario_id[1:]) % 4 != 0
    return {"outcomes": [-1 if worse else 0] * repeats, "fouls": 0, "stalls": 0, "errors": []}


def test_stop_at_t_ends_once_the_difference_is_clear(monkeypatch):
    monkeypatch.setattr(scenario_bench, "_runs", _fake_runs_candidate_worse)
    kwargs = dict(candidate="cand", opponent="opp", horizon_s=1.0, repeats=1, baseline="base")

    full, full_reason = scenario_bench._score(_many(60), **kwargs)
    stopped, reason = scenario_bench._score(_many(60), **kwargs, stop_at_t=4.0, check_every=10)

    assert len(full) == 60 and full_reason is None
    assert len(stopped) < 60 and len(stopped) % 10 == 0  # stops at a check, before the end
    assert reason == "detected"
    assert abs(scenario_bench._t([r["delta"] for r in stopped])) >= 4.0


def test_stop_at_t_scores_a_shuffled_sample_not_the_bank_s_first_starts(monkeypatch):
    # a bank is ordered by match and family: its first 20 starts are not a fair sample
    monkeypatch.setattr(scenario_bench, "_runs", _fake_runs_candidate_worse)
    kwargs = dict(candidate="cand", opponent="opp", horizon_s=1.0, repeats=1, baseline="base")

    first, _ = scenario_bench._score(_many(60), **kwargs, stop_at_t=4.0, check_every=10)
    again, _ = scenario_bench._score(_many(60), **kwargs, stop_at_t=4.0, check_every=10)

    ids = [r["scenario_id"] for r in first]
    assert ids == [r["scenario_id"] for r in again]  # reproducible
    assert ids != [f"s{i:03d}" for i in range(len(ids))]


def test_stop_at_t_stops_a_candidate_no_different_from_the_baseline_as_futile(monkeypatch):
    # the same outcomes on both sides never reach |t| >= 4; without futility this costs
    # the whole bank
    monkeypatch.setattr(
        scenario_bench,
        "_runs",
        lambda bench_scenario, config, opponent, horizon_s, repeats, *_reuse: {
            "outcomes": [0] * repeats,
            "fouls": 0,
            "stalls": 0,
            "errors": [],
        },
    )
    kwargs = dict(candidate="cand", opponent="opp", horizon_s=1.0, repeats=1, baseline="base")

    rows, reason = scenario_bench._score(_many(60), **kwargs, stop_at_t=4.0, check_every=10)

    assert (len(rows), reason) == (10, "futile")


def _spread(a: float, n: int = 200) -> list[float]:
    # mean 0, deltas +a and -a: the only thing deciding futility is the stderr
    return [a, -a] * (n // 2)


def test_futile_needs_the_upper_bound_of_the_difference_under_futile_below():
    # |mean| + FUTILE_Z * stderr == FUTILE_BELOW exactly at a = a_edge
    n = 200
    stderr_per_a = math.sqrt(n / (n - 1)) / math.sqrt(n)
    a_edge = scenario_bench.FUTILE_BELOW / (scenario_bench.FUTILE_Z * stderr_per_a)

    assert scenario_bench._stop_reason(_spread(a_edge * 0.99, n), stop_at_t=4.0) == "futile"
    assert scenario_bench._stop_reason(_spread(a_edge * 1.01, n), stop_at_t=4.0) is None
    # a real mean difference counts against futility, not just the noise
    shifted = [d + 0.05 for d in _spread(a_edge * 0.99, n)]
    assert scenario_bench._stop_reason(shifted, stop_at_t=4.0) is None


def test_a_clear_difference_is_detected_not_futile():
    deltas = [-1.0] * 30 + [0.0] * 70  # t far past 4
    assert scenario_bench._stop_reason(deltas, stop_at_t=4.0) == "detected"
    assert scenario_bench._stop_reason([0.0], stop_at_t=4.0) is None  # one delta decides nothing
