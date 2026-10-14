"""Tests for `utama_core.dashboard.views.tournament`'s `/tournament/runs` payload."""

from __future__ import annotations

import json

from utama_core.dashboard.views import tournament as tournament_view


def _write_run(root, results):
    run = root / "tournament_1"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"run_id": "tournament_1", "results": results}))
    return run


def test_each_match_carries_the_path_of_its_replay_when_the_file_exists(tmp_path, monkeypatch):
    # round_robin.py names a replay "<a>_vs_<b>.npz"; the dashboard must not guess
    # another runner's naming and link to a file that isn't there
    run = _write_run(
        tmp_path,
        [
            {"config_a": "build_low_block_kernel_strategy", "config_b": "build_three_slot_kernel_strategy"},
            {"config_a": "build_tiki_taka_kernel_strategy", "config_b": "build_counter_flow_kernel_strategy"},
        ],
    )
    (run / "low_block_vs_three_slot.npz").write_bytes(b"")
    monkeypatch.setattr(tournament_view, "REPLAY_BASE_PATH", tmp_path)

    (loaded,) = tournament_view._load_runs()

    assert [r["replay"] for r in loaded["results"]] == ["tournament_1/low_block_vs_three_slot.npz", None]


def test_full_match_tournament_replays_are_found_by_their_side_and_kickoff_tag(tmp_path, monkeypatch):
    run = _write_run(
        tmp_path,
        [
            {
                "config_a": "build_a_kernel_strategy",
                "config_b": "build_b_kernel_strategy",
                "a_is_right": False,
                "a_kicks_off": True,
            }
        ],
    )
    (run / "a_vs_b_LK.pkl").write_bytes(b"")
    monkeypatch.setattr(tournament_view, "REPLAY_BASE_PATH", tmp_path)

    (loaded,) = tournament_view._load_runs()

    assert loaded["results"][0]["replay"] == "tournament_1/a_vs_b_LK.pkl"
