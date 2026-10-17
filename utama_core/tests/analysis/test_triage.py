"""tools/triage.py without replays: which matches are chosen (each strategy's worst N as config_a, by goal
difference), the picture ranking and cap, and the index's text. Match files are empty stubs and their scores
come from a fake `final_score`, so nothing is simulated or drawn from a real replay."""

from pathlib import Path

import pytest

from tools import triage
from utama_core.analysis import matches as match_files
from utama_core.analysis.findings import Finding

A, B, C = "build_alpha_kernel_strategy", "build_beta_kernel_strategy", "build_gamma_kernel_strategy"


def _finding(kind, value, *, robots=(1,), t0=0.0, t1=10.0, message=None):
    return Finding(
        tool="test",
        kind=kind,
        replay=None,
        side="friendly",
        t0=t0,
        t1=t1,
        robots=robots,
        value=value,
        threshold=None,
        unit="s",
        message=message or f"{kind} {value}",
    )


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Stub match files (alpha as config_a in three matches, as config_b in one) and their scores."""
    d = tmp_path / "run"
    d.mkdir()
    scores = {
        "alpha_vs_beta": (0, 3),  # alpha GD -3
        "alpha_vs_gamma_s1": (2, 2),  # alpha GD 0
        "alpha_vs_gamma_s2": (1, 0),  # alpha GD +1
        "beta_vs_alpha": (4, 0),  # alpha is config_b here, GD -4 for alpha: must not count as alpha's config_a match
        "beta_vs_gamma": (1, 1),
    }
    for stem in scores:
        (d / f"{stem}.npz").write_bytes(b"")
    monkeypatch.setattr(match_files, "final_score", lambda npz: scores[npz.name.removesuffix(".npz")])
    return d


def test_worst_n_by_goal_difference_as_config_a(run_dir):
    chosen = triage.select_matches(run_dir, None, 2)
    alpha = [m.npz.name for m in chosen if m.a == "alpha"]
    assert alpha == ["alpha_vs_beta.npz", "alpha_vs_gamma_s1.npz"]  # GD -3 then 0; the +1 match is the best and cut


def test_strategy_filter_takes_full_or_short_names(run_dir):
    chosen = triage.select_matches(run_dir, {A}, 5)
    assert {m.a for m in chosen} == {"alpha"}
    assert len(chosen) == 3
    chosen = triage.select_matches(run_dir, {"beta"}, 5)
    assert [m.npz.name for m in chosen] == ["beta_vs_gamma.npz", "beta_vs_alpha.npz"]  # GD 0 then +4


def test_every_strategy_gets_its_own_worst_matches(run_dir):
    chosen = triage.select_matches(run_dir, None, 1)
    assert sorted((m.a, m.npz.name) for m in chosen) == [
        ("alpha", "alpha_vs_beta.npz"),
        ("beta", "beta_vs_gamma.npz"),
    ]


def test_rank_puts_kind_priority_first_then_value():
    items = [
        triage.Item(_finding("tactic_thrash", 50.0), png=Path("a.png")),
        triage.Item(_finding("stall", 3.0), png=Path("b.png")),
        triage.Item(_finding("tactic_thrash", 9.0), png=Path("c.png")),
        triage.Item(_finding("parked", 1.0), png=Path("d.png")),
        triage.Item(_finding("target_in_keep_out", 0.1), request=triage.RestartPicture(Path("r.npz"), 1.0, None)),
        triage.Item(_finding("stall", 9.0)),  # no picture source: never ranked
    ]
    ranked = triage.rank(items)
    assert [it.finding.kind for it in ranked] == [
        "stall",
        "target_in_keep_out",
        "parked",
        "tactic_thrash",
        "tactic_thrash",
    ]
    assert [it.finding.value for it in ranked if it.finding.kind == "tactic_thrash"] == [50.0, 9.0]


def test_draw_pictures_caps_at_k_and_shares_one_file(tmp_path):
    src = tmp_path / "plan.png"
    src.write_bytes(b"\x89PNG stub")
    items = [triage.Item(_finding("stall", 3.0), png=src)]
    items += [triage.Item(_finding("parked", 2.0 + i), png=src) for i in range(3)]
    items += [triage.Item(_finding("tactic_thrash", 5.0), png=src)]
    pictures = tmp_path / "pictures"
    pictures.mkdir()
    shown = triage.draw_pictures(items, pictures, max_pictures=2)
    assert len(shown) == 2
    pictured = [it for it in items if it.image is not None]
    assert [it.finding.kind for it in pictured] == ["stall", "parked"]
    assert len({it.image for it in pictured}) == 1  # the same plan_trace file for both
    assert pictured[0].image.read_bytes() == b"\x89PNG stub"


def test_index_has_each_pictured_finding_message_and_path(run_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    picture = out / "01_stall.png"
    picture.write_bytes(b"x")
    stall = triage.Item(
        _finding("stall", 3.0, message="R1 oscillates with the ball frozen for 3.0 s."),
        png=picture,
        robot=1,
        window=(783.0, 786.0),
        image=picture,
    )
    quiet = triage.Item(
        _finding("tactic_thrash", 4.0, message="R2 thrashes, no picture.")
    )  # not pictured: counted only
    m = triage.select_matches(run_dir, {A}, 1)[0]
    report = triage.MatchReport(m, [stall, quiet], notes=[])
    index = triage.write_index(out, run_dir, [report], max_pictures=8, restarts=False).read_text()
    assert "R1 oscillates with the ball frozen for 3.0 s." in index
    assert str(picture.resolve()) in index
    assert "Picture of 783.0-786.0 s (R1)" in index
    assert "R2 thrashes, no picture." not in index  # only pictured findings are described
    assert "| stall | 1 |" in index and "| tactic_thrash | 1 |" in index  # every finding is counted
    assert "alpha_vs_beta: 0-3 (GD -3)" in index


def test_index_says_so_when_a_strategy_has_no_matches(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    index = triage.write_index(out, tmp_path, [], max_pictures=8, restarts=False).read_text()
    assert "No matches with this strategy" in index
