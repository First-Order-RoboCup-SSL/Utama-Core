import json

import numpy as np

from tools.match_timeline import (
    OTHER,
    SLOTS,
    Span,
    goalkeeper_ids,
    group_spans,
    lane_summary,
    ref_group,
    render_timeline,
    runs,
    spans_from_changes,
    switch_times,
    tactic_changes,
    tactic_colours,
)


def _row(t, slot, robots, note):
    return {"sim_time": t, "tactic_id": slot, "robot_ids": robots, "note": note}


def _spans(rows, robot, t0, t1):
    changes = tactic_changes(rows, [robot])[robot]
    return spans_from_changes(changes, t0, t1)


def test_switching_a_to_b_to_a_gives_two_switches_and_spans_tile_the_window():
    rows = [
        _row(1.0, "attack", [1], "ClassA"),
        _row(2.0, "defense", [1], "ClassB"),
        _row(3.0, "attack", [1], "ClassA"),
    ]
    spans = _spans(rows, 1, 0.0, 10.0)
    assert [s.label for s in spans] == [None, "ClassA", "ClassB", "ClassA"]
    assert switch_times(spans) == [2.0, 3.0]
    assert lane_summary(spans, 0.0, 10.0)["switches"] == 2
    # Spans tile [0, 10] with no gap and no overlap.
    assert spans[0].t0 == 0.0 and spans[-1].t1 == 10.0
    assert all(a.t1 == b.t0 for a, b in zip(spans, spans[1:]))
    assert sum(s.t1 - s.t0 for s in spans) == 10.0


def test_switches_per_minute_use_the_window_length():
    rows = [_row(10.0, "attack", [2], "X"), _row(20.0, "defense", [2], "Y"), _row(30.0, "attack", [2], "X")]
    spans = _spans(rows, 2, 0.0, 60.0)
    summary = lane_summary(spans, 0.0, 60.0)
    assert summary["switches"] == 2
    assert summary["per_min"] == 2.0
    assert summary["longest"] == ("X", 30.0)  # X held 10-20 s and 30-60 s; the longest is 30-60 s


def test_window_clips_spans_and_only_counts_switches_inside_it():
    rows = [_row(1.0, "attack", [1], "A"), _row(5.0, "defense", [1], "B")]
    spans = _spans(rows, 1, 3.0, 4.0)
    assert spans == [Span(3.0, 4.0, "A")]
    assert switch_times(spans) == []


def test_unassigned_gap_between_the_same_tactic_is_not_a_switch():
    rows = [_row(1.0, "attack", [1], "A"), _row(2.0, "attack", [], "released"), _row(3.0, "attack", [1], "A")]
    spans = _spans(rows, 1, 0.0, 5.0)
    assert [s.label for s in spans] == [None, "A", None, "A"]
    assert switch_times(spans) == []


def test_tactic_name_comes_from_the_class_note_and_survives_a_deadline_release():
    rows = [
        _row(1.0, "attack", [1], "LeadAndSupportTactic"),
        _row(2.0, "attack", [1], "deadline release after 3.0s committed, ball moved 0.001m"),
    ]
    spans = _spans(rows, 1, 0.0, 4.0)
    assert [s.label for s in spans] == [None, "LeadAndSupportTactic"]


def test_referee_reset_clears_every_slot():
    rows = [
        _row(1.0, "attack", [1], "A"),
        _row(4.0, "referee reset", [], "referee restart cleared all tactic state"),
    ]
    spans = _spans(rows, 1, 0.0, 6.0)
    assert [s.label for s in spans] == [None, "A", None]


def test_robot_in_no_slot_stays_unassigned():
    rows = [_row(1.0, "attack", [2, 3], "A")]
    assert _spans(rows, 1, 0.0, 2.0) == [Span(0.0, 2.0, None)]


def test_runs_gives_contiguous_true_spans():
    flag = np.array([False, True, True, False, True, True])
    t = np.arange(6, dtype=float)
    assert runs(flag, t, "ball") == [Span(1.0, 3.0, "ball"), Span(4.0, 5.0, "ball")]


def test_ref_group_and_group_spans():
    assert ref_group("NORMAL_START") == "live play"
    assert ref_group("GOAL_BLUE") == "STOP"
    assert ref_group("DIRECT_FREE_YELLOW") == "free kick"
    assert ref_group("PREPARE_KICKOFF_BLUE") == "kickoff"
    assert ref_group("PREPARE_PENALTY_YELLOW") == "penalty"
    assert ref_group("BALL_PLACEMENT_BLUE") == "placement"
    assert ref_group("HALT") == "HALT"
    assert ref_group("-") == "no referee"
    groups = ["STOP", "STOP", "live play", "live play"]
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert group_spans(groups, t, 0.0, 4.0) == [Span(0.0, 2.0, "STOP"), Span(2.0, 4.0, "live play")]


def test_tactics_beyond_seven_slots_fold_into_other():
    lanes = [[Span(0.0, 10.0 - i, f"T{i}")] for i in range(8)]
    colours = tactic_colours(lanes)
    assert len(colours) == len(SLOTS) == 7
    assert "T7" not in colours  # the eighth, least-used tactic
    assert colours["T0"] == SLOTS[0]
    assert OTHER not in colours.values()


def test_goalkeeper_ids_read_the_highlights_trace(tmp_path):
    npz = tmp_path / "a_vs_b.npz"
    rows = [
        {"event": "trace", "sim_time": 0.0, "key": "highlights.goalkeeper", "value": {"0": "keeper"}},
        {"event": "trace", "sim_time": 1.0, "key": "other", "value": {"4": "x"}},
    ]
    (tmp_path / "a_vs_b.intentions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert goalkeeper_ids(npz) == {0}


def test_render_writes_a_png_with_every_panel(tmp_path):
    t = np.round(np.arange(0.0, 20.0, 0.1), 3)
    robot_ids = [1, 2]
    lanes = [
        [Span(0.0, 8.0, None), Span(8.0, 20.0, "LeadAndSupportTactic")],
        [Span(0.0, 10.0, "ShadowAndMarkTactic"), Span(10.0, 20.0, "LeadAndSupportTactic")],
    ]
    possession = [[Span(8.0, 9.0, "ball")], []]
    ref = [Span(0.0, 5.0, "live play"), Span(5.0, 20.0, "STOP")]
    dist = np.column_stack([np.linspace(6, 0.1, t.size), np.linspace(2, 1, t.size)])
    events = [
        {"kind": "GOAL", "t": 12.0, "side": "theirs", "text": "GOAL theirs"},
        {"kind": "FOUL", "t": 6.0, "side": "ours", "rule": "excessive_dribbling"},
        {"kind": "SHOT", "t": 9.0, "side": "ours"},
    ]
    out = tmp_path / "timeline.png"
    path = render_timeline(
        out,
        t0=0.0,
        t1=20.0,
        side="friendly",
        robot_ids=robot_ids,
        lanes=lanes,
        possession=possession,
        ref_spans=ref,
        t=t,
        dist=dist,
        events=events,
        keepers={1},
        title="test",
    )
    assert path == out
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 5_000
