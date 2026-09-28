import json

from utama_core.replay.list_runs import run_rows


def test_reads_summary_and_tolerates_missing_fields(tmp_path):
    full = tmp_path / "tournament_20260928_083032"
    full.mkdir()
    (full / "summary.json").write_text(
        json.dumps(
            {
                "run": {
                    "git_commit": "b55ec76fc2b2c4b9",
                    "git_dirty": True,
                    "argv": ["--max-workers", "6", "low_block"],
                    "started_utc": "2026-09-28T08:31:28+00:00",
                },
                "results": [{}, {}, {}],
                "stalled_match_count": 1,
            }
        )
    )
    old = tmp_path / "tournament_20260101_000000"
    old.mkdir()
    (old / "summary.json").write_text(json.dumps({"results": []}))
    (tmp_path / "tournament_20260929_000000").mkdir()  # in progress: no summary yet
    (tmp_path / "ab_cur").mkdir()

    rows = run_rows(tmp_path)

    assert [r["run"] for r in rows] == [
        "tournament_20260101_000000",
        "tournament_20260928_083032",
        "tournament_20260929_000000",
    ]
    assert rows[0] == {
        "run": "tournament_20260101_000000",
        "started": "-",
        "commit": "-",
        "matches": 0,
        "stalled": "-",
        "argv": "-",
    }
    assert rows[1] == {
        "run": "tournament_20260928_083032",
        "started": "2026-09-28T08:31:28+00:00",
        "commit": "b55ec76f*",
        "matches": 3,
        "stalled": 1,
        "argv": "--max-workers 6 low_block",
    }
    assert rows[2]["matches"] == "-"
