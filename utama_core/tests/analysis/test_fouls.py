"""`utama_core/analysis/fouls.py`: our side's fouls in a run with the referee command active, as
`tools/debug/find_bugs.py --fouls` lists them (moved from `tools/debug/moments.py fouls` on 2026-11-11)."""

import json

import pytest

from utama_core.analysis import fouls as F
from utama_core.entities.referee.referee_command import RefereeCommand


@pytest.mark.parametrize(
    "command, we_are_yellow, side",
    [
        ("DIRECT_FREE_YELLOW", True, "ours"),
        ("BALL_PLACEMENT_YELLOW", True, "ours"),
        ("DIRECT_FREE_BLUE", True, "theirs"),
        ("DIRECT_FREE_BLUE", False, "ours"),
        ("NORMAL_START", True, None),
        ("STOP", True, None),
        ("TIMEOUT_YELLOW", True, None),
    ],
)
def test_restart_side_is_ours_theirs_or_neither(command, we_are_yellow, side):
    assert F.restart_side(RefereeCommand[command], we_are_yellow) == side
    assert F.restart_side(None, we_are_yellow) is None


def _run(tmp_path, fouls, sidecar_rows=None):
    results = [{"config_a": "a", "config_b": "b", "stats": {"fouls": fouls}}]
    (tmp_path / "summary.json").write_text(json.dumps({"results": results}))
    if sidecar_rows is not None:
        (tmp_path / "a_vs_b.intentions.jsonl").write_text("\n".join(json.dumps(r) for r in sidecar_rows) + "\n")
    return tmp_path


def _foul(t, side="friendly", rule="keep_out"):
    return {"rule": rule, "sim_time": t, "side": side, "robot_id": 1, "tactic": "X", "inferred": False}


def test_our_fouls_names_the_command_active_and_the_restart_side(tmp_path):
    rows = [
        {"event": "referee", "sim_time": 0.0, "command": "NORMAL_START"},
        {"event": "referee", "sim_time": 4.0, "command": "DIRECT_FREE_BLUE"},
        {"event": "referee", "sim_time": 10.0, "command": "DIRECT_FREE_YELLOW"},
    ]
    run = _run(tmp_path, [_foul(1.0), _foul(5.0), _foul(12.0), _foul(20.0), _foul(6.0, side="enemy")], rows)
    found = F.our_fouls(run)
    assert [(f["sim_time"], f["command"], f["restart"]) for f in found] == [
        (1.0, "NORMAL_START", None),
        (5.0, "DIRECT_FREE_BLUE", "theirs"),
        (12.0, "DIRECT_FREE_YELLOW", "ours"),
        (20.0, "DIRECT_FREE_YELLOW", "ours"),
    ]
    assert [f["sim_time"] for f in F.without_own_restarts(found)] == [1.0, 5.0]


def test_our_fouls_names_the_match_replay_relative_to_the_repo(tmp_path):
    run = _run(tmp_path, [_foul(3.0)])
    (found,) = F.our_fouls(run)
    assert found["replay"] is None  # no a_vs_b replay in the run dir
    (tmp_path / "a_vs_b.npz").write_bytes(b"")
    (found,) = F.our_fouls(run)
    assert found["replay"] == str(tmp_path / "a_vs_b.npz")  # outside the repo: as given
    (tmp_path / "a_vs_b.npz").unlink()
    (tmp_path / "a_vs_b.pkl").write_bytes(b"")
    (found,) = F.our_fouls(run)
    assert found["replay"] == str(tmp_path / "a_vs_b.pkl")


def test_repo_relative_paths_drop_the_repo_root():
    assert F._repo_relative(F.REPO_ROOT / "replays" / "t" / "a_vs_b.npz") == "replays/t/a_vs_b.npz"


def test_our_fouls_without_a_match_log_has_no_command(tmp_path):
    (found,) = F.our_fouls(_run(tmp_path, [_foul(3.0)]))
    assert found["command"] is None and found["restart"] is None
    assert F.without_own_restarts([found]) == [found]


def _row(config_a, rule, restart=None, tactic="X"):
    return {"config_a": config_a, "config_b": "b", "rule": rule, "tactic": tactic, "restart": restart}


def test_count_fouls_groups_by_the_given_columns_most_common_first():
    rows = [_row("a", "keep_out"), _row("b", "crashing"), _row("a", "keep_out")]
    rows += [_row("a", "crashing"), _row("b", "crashing"), _row("b", "crashing")]
    assert F.count_fouls(rows, ["strategy", "rule"]) == [
        (("b", "crashing"), 3),
        (("a", "keep_out"), 2),
        (("a", "crashing"), 1),
    ]


def test_count_fouls_names_a_missing_restart_live_and_a_missing_tactic_dash():
    rows = [_row("a", "keep_out", restart=None, tactic=None), _row("a", "keep_out", restart="theirs")]
    assert F.count_fouls(rows, ["restart", "tactic"]) == [(("live", "-"), 1), (("theirs", "X"), 1)]


def test_count_fouls_ties_are_in_name_order():
    assert [k for k, _ in F.count_fouls([_row("b", "keep_out"), _row("a", "keep_out")], ["strategy"])] == [
        ("a",),
        ("b",),
    ]


def test_without_stoppages_drops_out_of_bounds_and_keeps_rules_fouls():
    rows = [_row("a", "out_of_bounds"), _row("a", "defense_area_stoppage"), _row("a", "keep_out")]
    assert [f["rule"] for f in F.without_stoppages(rows)] == ["defense_area_stoppage", "keep_out"]


def test_fouls_by_table_has_header_counts_and_total():
    lines = F.fouls_by_table([_row("a", "keep_out"), _row("a", "keep_out")], ["strategy", "rule"])
    assert [line.split() for line in lines] == [["count", "strategy", "rule"], ["2", "a", "keep_out"], ["2", "total"]]


def test_fouls_json_is_the_foul_list_or_the_by_counts_as_rows():
    rows = [_row("a", "keep_out", restart=None), _row("a", "keep_out"), _row("b", "crashing")]
    assert json.loads(F.fouls_json(rows)) == rows
    assert json.loads(F.fouls_json(rows, ["strategy", "rule"])) == [
        {"strategy": "a", "rule": "keep_out", "count": 2},
        {"strategy": "b", "rule": "crashing", "count": 1},
    ]


def test_find_bugs_fouls_filters_then_counts(tmp_path):
    from tools.debug.find_bugs import fouls_report

    run = _run(tmp_path, [_foul(1.0), _foul(2.0, rule="out_of_bounds"), _foul(3.0, rule="crashing")])
    assert len(fouls_report(run)) == 2  # out_of_bounds is a stoppage, left out
    assert len(fouls_report(run, include_stoppages=True)) == 3
    assert fouls_report(run, rule="crashing")[0].startswith("crashing")
    assert [line.split() for line in fouls_report(run, by="rule")][-1] == ["2", "total"]
    with pytest.raises(ValueError, match="unknown column"):
        fouls_report(run, by="nope")
