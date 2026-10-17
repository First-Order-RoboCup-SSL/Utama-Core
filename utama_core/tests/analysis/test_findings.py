"""The shared Finding format: JSON round trip and the stdout path (no replay, no simulator)."""

import json

from utama_core.analysis.findings import (
    Finding,
    from_dict,
    read_jsonl,
    to_dict,
    write_jsonl,
)


def _finding(**overrides) -> Finding:
    fields = dict(
        tool="match_timeline",
        kind="tactic_thrash",
        replay="replays/run/a_vs_b.npz",
        side="friendly",
        t0=0.0,
        t1=480.0,
        robots=(4, 5),
        value=6.5,
        threshold=4.0,
        unit="per live min",
        message="R4 changed tactic 13 times in 2.0 min of live play (6.50 per min, threshold 4).",
        image=None,
    )
    fields.update(overrides)
    return Finding(**fields)


def test_jsonl_round_trip_keeps_every_field(tmp_path):
    findings = [
        _finding(),
        _finding(kind="zone_share", replay=None, side=None, t0=None, t1=None, robots=(), threshold=None, image="a.png"),
    ]
    path = tmp_path / "out.jsonl"
    write_jsonl(findings, path)

    assert read_jsonl(path) == findings
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["robots"] == [4, 5]


def test_to_dict_lists_robots_and_from_dict_restores_a_tuple():
    d = to_dict(_finding(robots=(7,)))
    assert d["robots"] == [7] and json.dumps(d)  # JSON-serialisable
    assert from_dict(d).robots == (7,)


def test_write_jsonl_to_stdout_with_dash(capsys):
    write_jsonl([_finding()], "-")
    out = capsys.readouterr().out
    assert json.loads(out.strip())["kind"] == "tactic_thrash"


def test_an_empty_list_writes_an_empty_file(tmp_path):
    path = tmp_path / "none.jsonl"
    write_jsonl([], path)
    assert path.read_text() == "" and read_jsonl(path) == []
