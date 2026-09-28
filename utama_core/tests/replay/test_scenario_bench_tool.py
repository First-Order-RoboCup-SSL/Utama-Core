import dataclasses
import importlib.util
import json
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


def test_growing_a_bank_screens_only_new_scenarios_and_keeps_the_informative_ones(tmp_path, monkeypatch):
    existing = list(all_hand_authored_scenarios())
    old_bank, new_bank = tmp_path / "bank_old.json", tmp_path / "bank_new.json"
    save_bank(existing, old_bank, bank_id="old")
    harvested = [
        _moved(existing[0], "copy_of_existing", 0.01),
        _moved(existing[0], "informative", 1.0),
        _moved(existing[1], "dead", 1.0),
    ]
    monkeypatch.setattr(scenario_bench, "harvest_run_dir", lambda *a, **k: (harvested, {}))
    screened = []

    def fake_screen(bench_scenario, *, horizon_s, repeats):
        screened.append(bench_scenario.scenario_id)
        verdict = "dead" if bench_scenario.scenario_id == "dead" else "informative"
        return {
            "scenario_id": bench_scenario.scenario_id,
            "verdict": verdict,
            "outcomes": [],
            "outcome_stdev": 0.0,
            "seed_stdev": 0.0,
            "policy_stdev": 0.0,
        }

    monkeypatch.setattr(scenario_bench, "_screen_one", fake_screen)
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
            "--dynamic-screen",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert scenario_bench.main() == 0

    # the hand-authored scenarios the tool always loads, and the near-copy, duplicate the old bank
    assert screened == ["informative", "dead"]
    bank_id, saved = load_bank(new_bank)
    assert bank_id == "bank_new"
    assert [s.scenario_id for s in saved] == [s.scenario_id for s in existing] + ["informative"]


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

    def fake_runs(bench_scenario, config, opponent, horizon_s, repeats):
        failed = bench_scenario.scenario_id == broken.scenario_id and config == "base"
        return {"outcomes": [0] * repeats, "fouls": 0, "stalls": 0, "errors": ["setup failed"] if failed else []}

    monkeypatch.setattr(scenario_bench, "_runs", fake_runs)
    rows = scenario_bench._score(
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
