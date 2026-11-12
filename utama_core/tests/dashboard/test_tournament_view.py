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
    # evaluate.py names a replay "<a>_vs_<b>.npz"; the dashboard must not guess
    # another runner's naming and link to a file that isn't there
    run = _write_run(
        tmp_path,
        [
            {"config_a": "low_block", "config_b": "three_slot"},
            {"config_a": "tiki_taka", "config_b": "riptide"},
        ],
    )
    (run / "low_block_vs_three_slot.npz").write_bytes(b"")
    monkeypatch.setattr(tournament_view, "REPLAY_BASE_PATH", tmp_path)

    (loaded,) = tournament_view._load_runs()

    assert [r["replay"] for r in loaded["results"]] == ["tournament_1/low_block_vs_three_slot.npz", None]


def test_replays_with_side_and_kickoff_tags_are_found(tmp_path, monkeypatch):
    run = _write_run(
        tmp_path,
        [
            {
                "config_a": "a",
                "config_b": "b",
                "a_is_right": False,
                "a_kicks_off": True,
            }
        ],
    )
    (run / "a_vs_b_LK.pkl").write_bytes(b"")
    monkeypatch.setattr(tournament_view, "REPLAY_BASE_PATH", tmp_path)

    (loaded,) = tournament_view._load_runs()

    assert loaded["results"][0]["replay"] == "tournament_1/a_vs_b_LK.pkl"
