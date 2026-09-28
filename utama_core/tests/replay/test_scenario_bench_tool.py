import dataclasses
import importlib.util
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
